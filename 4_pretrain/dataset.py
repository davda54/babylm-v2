import torch
import gzip


def apply_mask(args, input_ids, mask_ratios, replacement_ids, global_step):
    mask_p = args.mask_p_start + (args.mask_p_end - args.mask_p_start) * global_step / args.max_steps
    mask_p = max(mask_p, mask_ratios.min().item())

    mask = mask_ratios < mask_p
    target_ids = torch.where(mask, input_ids, -100)
    input_ids = torch.where(mask, replacement_ids, input_ids)

    real_mask_p = mask.sum().item() / mask_ratios.numel()

    return input_ids, target_ids, real_mask_p


class SpanMaskingStrategy:
    def __init__(self, n_special_tokens, random_p, keep_p, vocab_size, mask_token_id):
        self.n_special_tokens = n_special_tokens
        self.random_p = random_p
        self.keep_p = keep_p
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.max_span_length = 3

    def __call__(self, tokens):
        replacement_tokens = tokens.clone()
        length = tokens.size(0)

        preservation_mask = tokens < self.n_special_tokens

        span_lengths = torch.randint(1, self.max_span_length + 1, [length]).long()

        indices = torch.repeat_interleave(torch.arange(span_lengths.size(0)), span_lengths)
        indices = indices[:length]
        if indices.size(0) < length:
            indices = torch.cat([indices, torch.full([length - indices.size(0)], fill_value=length // 2 - 1, dtype=torch.long)])

        max_index = indices[-1].item()
        span_random_numbers_1, span_random_numbers_2 = torch.rand([(max_index + 1) * 2]).chunk(2)
        
        mask_ratios = span_random_numbers_1[indices]
        mask_ratios[preservation_mask] = 1.0

        replacement_p = span_random_numbers_2[indices]
        random_mask = replacement_p < self.random_p
        replacement_tokens[random_mask] = torch.randint(
            low=self.n_special_tokens,
            high=self.vocab_size,
            size=[random_mask.sum().item()],
            dtype=torch.long
        )
        replacement_tokens[replacement_p > (self.random_p + self.keep_p)] = self.mask_token_id

        return mask_ratios, replacement_tokens


class RandomIndex:
    def __init__(self, n_segments):
        self.n_segments = n_segments
        self.indices = torch.randperm(n_segments)
        self.index = 0

    def get_random_index(self):
        if self.index >= self.n_segments:
            self.indices = torch.randperm(self.n_segments)
            self.index = 0

        index = self.indices[self.index]
        self.index += 1

        return index


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_file: str, tokenizer, args, count_used=False):
        self.path = input_file
        self.count_used = count_used
        self.seq_length = args.seq_length
        self.n_special_tokens = args.n_special_tokens

        self.mask_index = tokenizer.token_to_id("␥")
        self.cls_index = tokenizer.token_to_id("␂")
        self.sep_index = tokenizer.token_to_id("␃")
        self.pad_index = tokenizer.token_to_id("␢")

        self.masking_strategy = SpanMaskingStrategy(args.n_special_tokens, args.mask_random_p, args.mask_keep_p, args.vocab_size, self.mask_index)

        documents = torch.load(input_file)
        self.segments = [
            document[offset : offset + self.seq_length - 2]
            for document in documents
            for offset in range(0, len(document), self.seq_length - 2)
            if len(document) > 0 and len(document) - offset > 1
        ]

        if self.count_used:
            self.counts = [
                torch.zeros_like(segment)
                for segment in self.segments
            ]
        
        self.random_index = RandomIndex(len(self.segments))

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        tokens = self.segments[index]
        seq_length = min(self.seq_length - 2, tokens.size(0))

        segment = torch.cat([
            torch.LongTensor([self.cls_index]),
            tokens[:seq_length].long(),
            torch.LongTensor([self.sep_index])
        ])
        attention_mask = torch.ones(seq_length + 2, seq_length + 2, dtype=torch.bool)

        if self.count_used:
            self.counts[index][:seq_length] += 1

        while self.seq_length - 2 - segment.size(0) > 1:
            index = self.random_index.get_random_index()
            tokens = self.segments[index].long()
            seq_length = min(self.seq_length - 2 - segment.size(0), tokens.size(0))

            # select random offset
            offset = 0
            if seq_length < tokens.size(0):
                conv_weight = torch.ones(1, 1, seq_length)
                summed_counts = torch.nn.functional.conv1d(
                    self.counts[index].view(1, 1, -1).float(),
                    conv_weight
                ).squeeze()
                offset_probabilities = torch.nn.functional.softmax(
                    -summed_counts.float(), dim=0
                )
                offset = torch.multinomial(offset_probabilities, 1).item()

            tokens = tokens[offset:offset + seq_length]

            segment = torch.cat([
                segment,
                torch.LongTensor([self.cls_index]),
                tokens,
                torch.LongTensor([self.sep_index])
            ])
            attention_mask = torch.block_diag(
                attention_mask,
                torch.ones(seq_length + 2, seq_length + 2, dtype=torch.bool)
            )

            if self.count_used:
                self.counts[index][offset:offset+seq_length] += 1

        padding_length = self.seq_length - segment.size(0)
        if padding_length > 0:
            segment = torch.cat([
                segment,
                torch.LongTensor([self.pad_index] * padding_length)
            ])
            attention_mask = torch.block_diag(
                attention_mask,
                torch.zeros(padding_length, padding_length, dtype=torch.bool)
            )

        if self.count_used:
            return segment

        mask_ratios, replacement_tokens = self.masking_strategy(segment)

        return segment, attention_mask, mask_ratios, replacement_tokens

    def show_random_item(self, tokenizer):
        inputs, _, mask_ratios, replacement_tokens = self.__getitem__(torch.randint(0, len(self), []).item())
        print(' '.join(tokenizer.decode([i], skip_special_tokens=False) for i in inputs.tolist()), flush=True)
        print(' '.join(str(i) for i in inputs.tolist()), flush=True)
        print(' '.join(tokenizer.decode([i], skip_special_tokens=False) for i in replacement_tokens.tolist()), flush=True)
        print(mask_ratios, flush=True)


# class ValidationDataset(torch.utils.data.Dataset):
#     def __init__(self, input_file: str, tokenizer, device_index, n_devices, args):
#         self.path = input_file
#         self.seq_length = 128
#         self.n_special_tokens = args.n_special_tokens

#         self.mask_index = tokenizer.token_to_id("[MASK]")
#         self.cls_index = tokenizer.token_to_id("[CLS]")
#         self.sep_index = tokenizer.token_to_id("[SEP]")
#         self.pad_index = tokenizer.token_to_id("[PAD]")

#         self.masking_strategy = SpanMaskingStrategy(args.n_special_tokens, args.mask_random_p, args.mask_keep_p, args.vocab_size, self.mask_index)

#         with gzip.GzipFile(input_file, 'rb') as f:
#             documents = torch.load(f)

#         self.segments = [
#             document[offset : offset + self.seq_length - 2]
#             for document in documents
#             for offset in range(0, len(document), self.seq_length - 2)
#             if len(document) > 0
#         ]
#         self.segments = self.segments[:len(self.segments) // n_devices * n_devices]
#         self.segments = self.segments[device_index::n_devices]

#     def __len__(self):
#         return len(self.segments)

#     def __getitem__(self, index):
#         tokens = self.segments[index]

#         target_seq_length = self.seq_length - 2
#         tokens = tokens[:target_seq_length].long()

#         padding_length = (self.seq_length - 2) - tokens.size(0)
#         segment = torch.cat([
#             torch.LongTensor([self.cls_index]),
#             tokens,
#             torch.LongTensor([self.sep_index]),
#             torch.LongTensor([self.pad_index] * padding_length)
#         ])

#         attention_mask = torch.cat([
#             torch.zeros(len(tokens) + 2, dtype=torch.bool),
#             torch.ones(padding_length, dtype=torch.bool)
#         ])

#         mask_ratios, replacement_tokens = self.masking_strategy(segment)

#         return segment, attention_mask, mask_ratios, replacement_tokens


if __name__ == '__main__':
    from tokenizers import Tokenizer
    from tqdm import tqdm
    import argparse
    from collections import Counter
    import matplotlib.pyplot as plt

    tokenizer = Tokenizer.from_file('../tokenizer_10M.json')

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--seq_length", type=int, default=512)
        parser.add_argument("--short_p", type=float, default=0.0)
        parser.add_argument("--n_special_tokens", type=int, default=16)
        parser.add_argument("--mask_random_p", type=float, default=0.15)
        parser.add_argument("--mask_keep_p", type=float, default=0.1)
        parser.add_argument("--vocab_size", type=int, default=16_384)
        parser.add_argument("--n_iterations", type=int, default=100)
        return parser.parse_args()

    args = parse_args()
    dataset = Dataset("../data/train_10M_tokenized.bin", tokenizer, args, count_used=True)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8_192, num_workers=0, shuffle=True)

    for _ in tqdm(range(args.n_iterations)):
        for batch in data_loader:
            pass
    
    counter = Counter()
    for segment in dataset.counts:
        counter.update(segment.tolist())

    # show histogram of how many times each token was used, use reasonable bins
    plt.bar(counter.keys(), counter.values())
    plt.xlabel('Categories')
    plt.ylabel('Counts')
    plt.title('Histogram from Counter')
    plt.savefig('histogram_summed_weighted_offset.pdf')

    for k, v in counter.most_common():
        print(f"How many tokens were used {k} times: {v}")

    print(f"Number of segments: {len(dataset)}")
