import torch
import gzip
import random
from collections import Counter


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

    def __call__(self, tokens, counts, verbose=False):
        length = tokens.size(0)

        span_lengths = torch.randint(1, self.max_span_length + 1, size=(length,), dtype=torch.int)
        cumsum = torch.cumsum(span_lengths, dim=0)
        
        total_length = cumsum[-1].item()
        indices = torch.zeros(total_length, dtype=torch.int)
        indices[cumsum - span_lengths] = torch.arange(length, dtype=torch.int)
        indices = torch.cummax(indices, dim=0)[0]
        indices = indices[:length]

        span_lengths = torch.randint(
            low=1,
            high=self.max_span_length + 1,
            size=[length],
            dtype=torch.long
        )
        indices = torch.repeat_interleave(
            torch.arange(span_lengths.size(0)),
            span_lengths
        )
        indices = indices[:length]

        max_index = indices[-1].item()
        span_random_numbers_1, span_random_numbers_2 = torch.rand([(max_index + 1) * 2]).chunk(2)
        
        mask_ratios = span_random_numbers_1[indices]
        if verbose: print(mask_ratios[:10])

        counts = counts.float()
        counts[tokens < self.n_special_tokens] = float('-inf')
        if verbose: print(counts[:10])
        counts_p = torch.nn.functional.softmax(counts, dim=0)
        if verbose: print(counts_p[:10])
        mask_ratios = mask_ratios * counts_p
        if verbose: print(mask_ratios[:10])
        mask_ratios = mask_ratios / (mask_ratios.mean() * 2)
        mask_ratios[tokens < self.n_special_tokens] = 1.0
        if verbose: print(mask_ratios[:10])
        if verbose: print(flush=True)

        replacement_p = span_random_numbers_2[indices]
        random_mask = replacement_p < self.random_p

        replacement_tokens = tokens.clone()
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
    def __init__(self, input_file: str, tokenizer, args, stride=False, count_used=False):
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
        if stride:
            self.segments = self.segments[args.rank::args.world_size]
            random.seed(args.rank)
            random.shuffle(self.segments)

        if self.count_used:
            self.counts = [
                torch.zeros_like(segment)
                for segment in self.segments
            ]
            self.mask_counts = [
                torch.zeros_like(segment)
                for segment in self.segments
            ]
            self.mask_length_counter = Counter()
        
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

            mask_ratios, replacement_tokens = self.masking_strategy(tokens[:seq_length].long(), self.mask_counts[index][:seq_length], verbose=index==0)

            mask_p = 0.15
            masked_tokens = mask_ratios < mask_p
            self.mask_counts[index][:seq_length][masked_tokens] += 1

            mask_length = 0
            for i in range(masked_tokens.size(0)):
                if masked_tokens[i]:
                    mask_length += 1
                elif mask_length > 0:
                    self.mask_length_counter.update([mask_length])
                    mask_length = 0
            if mask_length > 0:
                self.mask_length_counter.update([mask_length])

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
                offset = torch.argmin(summed_counts)

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

                mask_ratios, replacement_tokens = self.masking_strategy(tokens, self.mask_counts[index][offset:offset+seq_length])

                mask_p = 0.15
                masked_tokens = mask_ratios < mask_p
                self.mask_counts[index][offset:offset+seq_length][masked_tokens] += 1

                mask_length = 0
                for i in range(masked_tokens.size(0)):
                    if masked_tokens[i]:
                        mask_length += 1
                    elif mask_length > 0:
                        self.mask_length_counter.update([mask_length])
                        mask_length = 0
                if mask_length > 0:
                    self.mask_length_counter.update([mask_length])

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
            return segment, attention_mask

        return segment, attention_mask, mask_ratios, replacement_tokens

    def show_random_item(self, tokenizer):
        inputs, _, mask_ratios, replacement_tokens = self.__getitem__(torch.randint(0, len(self), []).item())
        print(' '.join(tokenizer.decode([i], skip_special_tokens=False) for i in inputs.tolist()), flush=True)
        print(' '.join(str(i) for i in inputs.tolist()), flush=True)
        print(' '.join(tokenizer.decode([i], skip_special_tokens=False) for i in replacement_tokens.tolist()), flush=True)
        print(mask_ratios, flush=True)


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
    plt.title('Frequency of each token being used')
    plt.savefig('histogram_min_sum_offset.pdf')
    plt.close()

    for k, v in counter.most_common():
        print(f"How many tokens were used {k} times: {v}")

    print(f"Number of segments: {len(dataset)}")

    # show histogram of masking frequencies
    mask_counter = Counter()
    for segment in dataset.mask_counts:
        mask_counter.update(segment.tolist())
    
    plt.bar(mask_counter.keys(), mask_counter.values())
    plt.xlabel('Categories')
    plt.ylabel('Counts')
    plt.title('Frequency of masking each token')
    plt.savefig('histogram_min_masking_frequency_weighted_mask.pdf')
    plt.close()

    for k, v in mask_counter.most_common():
        print(f"How many tokens were masked {k} times: {v}")

    # show histogram of mask lengths
    plt.bar(dataset.mask_length_counter.keys(), dataset.mask_length_counter.values())
    plt.xlabel('Categories')
    plt.ylabel('Counts')
    plt.title('Frequency of mask lengths')
    plt.savefig('histogram_min_mask_length_weighted_mask.pdf')
