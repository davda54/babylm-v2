import torch
from torch.utils.data import Dataset
from smart_open import open
import random


class SpanMaskingStrategy:
    def __init__(self, mask_p, tokenizer, n_special_tokens, padding_label_id=-100, random_p=0.1, keep_p=0.1):
        self.mask_p = mask_p
        self.random_p = random_p
        self.keep_p = keep_p
        self.tokenizer = tokenizer
        self.n_special_tokens = n_special_tokens
        self.padding_label_id = padding_label_id
        self.mask_index = self.tokenizer.token_to_id("[MASK]")

    def __call__(self, tokens, mlm_length=None):
        labels = torch.full_like(tokens, fill_value=self.padding_label_id)
        inputs = tokens.clone()
        
        if mlm_length is None:
            n_masked = torch.binomial((tokens >= self.n_special_tokens).float().sum(dim=0, keepdim=True), torch.FloatTensor([self.mask_p])).item()
            preservation_mask = tokens < self.n_special_tokens
            mask = torch.zeros_like(tokens, dtype=torch.bool)
            counter = 100

            while n_masked > mask.long().sum() and counter > 0:
                span_length = torch.tensor([0]).geometric_(1/3).item() % 10
                offset = torch.randint(-(span_length - 1), tokens.size(0) + span_length, []).item()
                sub_mask = torch.zeros_like(tokens, dtype=torch.bool)
                sub_mask[max(0, offset) : min(mask.size(0)-1, offset + span_length)] = True
                sub_mask[preservation_mask] = False

                random_p = torch.rand([]).item()

                # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
                if random_p < 1.0 - self.random_p - self.keep_p:
                    inputs[sub_mask] = self.mask_index
                elif random_p < 1.0 - self.keep_p:
                    random_words = torch.randint(
                        low=self.n_special_tokens - 1,
                        high=self.tokenizer.get_vocab_size(),
                        size=(sub_mask.sum(),),
                        dtype=torch.long
                    )
                    inputs[sub_mask] = random_words
                else:
                    inputs[sub_mask] = tokens[sub_mask]

                mask |= sub_mask
                counter -= 1

            labels[mask] = tokens[mask]

        else:
            n_masked = torch.binomial((tokens[:mlm_length] >= self.n_special_tokens).float().sum(dim=0, keepdim=True), torch.FloatTensor([self.mask_p])).item()
            preservation_mask = tokens < self.n_special_tokens
            mask = torch.zeros_like(tokens, dtype=torch.bool)
            counter = 100

            while n_masked > mask.long().sum() and counter > 0:
                span_length = torch.tensor([0]).geometric_(1/3).item() % 10
                offset = torch.randint(-(span_length - 1), mlm_length + span_length, []).item()
                sub_mask = torch.zeros_like(tokens, dtype=torch.bool)
                sub_mask[max(0, offset) : min(mlm_length-1, offset + span_length)] = True
                sub_mask[preservation_mask] = False

                random_p = torch.rand([]).item()

                # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
                if random_p < 1.0 - self.random_p - self.keep_p:
                    inputs[sub_mask] = self.mask_index
                elif random_p < 1.0 - self.keep_p:
                    random_words = torch.randint(
                        low=self.n_special_tokens - 1,
                        high=self.tokenizer.get_vocab_size(),
                        size=(sub_mask.sum(),),
                        dtype=torch.long
                    )
                    inputs[sub_mask] = random_words
                else:
                    inputs[sub_mask] = tokens[sub_mask]

                mask |= sub_mask
                counter -= 1

            mask[mlm_length:] = True

            labels[mask] = tokens[mask]

        return inputs, labels, mask


class Dataset(Dataset):
    def __init__(self, file, offset: int, n_gpus: int, tokenizer, seq_length=512, mask_p=0.15, short_p=0.1, random_p=0.1, keep_p=0.1):
        self.tokenizer = tokenizer

        self.seq_length = seq_length
        self.short_p = short_p
        self.n_special_tokens = 6

        self.masking_strategy = SpanMaskingStrategy(mask_p, tokenizer, self.n_special_tokens, padding_label_id=-100, random_p=random_p, keep_p=keep_p)

        self.mask_index = self.tokenizer.token_to_id("[MASK]")
        self.cls_index = self.tokenizer.token_to_id("[CLS]")
        self.sep_index = self.tokenizer.token_to_id("[SEP]")
        self.pad_index = self.tokenizer.token_to_id("[PAD]")

        self.segments = []
        for i, segment in enumerate(open(file)):
            if i % n_gpus != offset:
                continue

            segment = segment.strip().split(" ")
            assert len(segment) <= seq_length - 2, " ".join(segment)
            segment = [self.tokenizer.token_to_id(token) for token in segment]
            self.segments.append(segment)

    def __len__(self):
        return len(self.segments)
    
    def rand(self):
        return torch.rand(1).item()

    def randint(self, low, high):
        return torch.randint(low=low, high=high, size=(1,)).item()

    def __getitem__(self, index):
        tokens = self.segments[index]

        target_seq_length = self.seq_length - 2 if self.rand() > self.short_p else self.randint(1, self.seq_length - 2)
        tokens = tokens[:target_seq_length]
        padding_length = (self.seq_length - 2) - len(tokens)
        segment = [self.cls_index] + tokens + [self.sep_index] + [self.pad_index] * padding_length
        segment = torch.LongTensor(segment)
        segment2 = [self.cls_index] + random.sample(tokens, len(tokens)) + [self.sep_index] + [self.pad_index] * padding_length
        segment2 = torch.LongTensor(segment2)

        attention_mask = torch.cat([
            torch.zeros(len(tokens) + 2, dtype=torch.bool),
            torch.ones(padding_length, dtype=torch.bool)
        ])

        inputs, outputs, seg_mask = self.masking_strategy(segment)

        seg_mask[len(tokens)+2:] = torch.ones(padding_length, dtype=torch.bool)

        return inputs, attention_mask, outputs

    def show_random_item(self):
        inputs, _, outputs = self.__getitem__(self.randint(0, len(self)))
        print(' '.join(self.tokenizer.id_to_token(i) for i in inputs), flush=True)
        print(' '.join(self.tokenizer.id_to_token(o) if o >= 0 else "-1" for o in outputs), flush=True)

class DatasetMirror(Dataset):
    def __init__(self, file, mirror_file, offset: int, n_gpus: int, tokenizer, seq_length=512, mirror_length=128, mask_p=0.15, short_p=0.1, random_p=0.1, keep_p=0.1):
        self.tokenizer = tokenizer

        self.seq_length = seq_length
        self.mirror_length = mirror_length
        self.short_p = short_p
        self.n_special_tokens = 6

        self.masking_strategy = SpanMaskingStrategy(mask_p, tokenizer, self.n_special_tokens, padding_label_id=-100, random_p=random_p, keep_p=keep_p)

        self.mask_index = self.tokenizer.token_to_id("[MASK]")
        self.cls_index = self.tokenizer.token_to_id("[CLS]")
        self.sep_index = self.tokenizer.token_to_id("[SEP]")
        self.pad_index = self.tokenizer.token_to_id("[PAD]")

        self.segments = []
        for i, segment in enumerate(open(file)):
            if i % n_gpus != offset:
                continue

            segment = segment.strip().split(" ")
            assert len(segment) <= seq_length - 2, " ".join(segment)
            segment = [self.tokenizer.token_to_id(token) for token in segment]
            self.segments.append(segment)

        self.mirror_segments = []
        for i, segment in enumerate(open(mirror_file)):
            if i % n_gpus != offset:
                continue

            segment = segment.strip().split(" ")
            assert len(segment) <= mirror_length - 2, " ".join(segment)
            segment = [self.tokenizer.token_to_id(token) for token in segment]
            self.mirror_segments.append(segment)

    def __len__(self):
        return len(self.segments)
    
    def rand(self):
        return torch.rand(1).item()

    def randint(self, low, high):
        return torch.randint(low=low, high=high, size=(1,)).item()

    def __getitem__(self, index):
        tokens = self.segments[index]
        select_mirror = random.randint(0, 9)
        mirror_tokens = self.mirror_segments[10*index + select_mirror]

        target_seq_length = self.seq_length - 2 if self.rand() > self.short_p else self.randint(1, self.seq_length - 2)
        tokens = tokens[:target_seq_length]
        padding_length = (self.seq_length - 2) - len(tokens)
        segment = [self.cls_index] + tokens + [self.sep_index] + [self.pad_index] * padding_length
        segment = torch.LongTensor(segment)

        padding_length2 = (self.mirror_length - 2) - len(mirror_tokens)
        segment2 = [self.cls_index] + mirror_tokens + [self.sep_index] + [self.pad_index] * padding_length2
        segment2 = torch.LongTensor(segment2)

        attention_mask = torch.cat([
            torch.zeros(len(tokens) + 2, dtype=torch.bool),
            torch.ones(padding_length, dtype=torch.bool)
        ])

        attention_mask2 = torch.cat([
            torch.zeros(len(mirror_tokens) + 2, dtype=torch.bool),
            torch.ones(padding_length2, dtype=torch.bool)
        ])

        inputs, outputs, seg_mask = self.masking_strategy(segment)

        seg_mask[len(tokens)+2:] = torch.ones(padding_length, dtype=torch.bool)

        return inputs, attention_mask, outputs, segment2, attention_mask2

    def show_random_item(self):
        inputs, _, outputs = self.__getitem__(self.randint(0, len(self)))
        print(' '.join(self.tokenizer.id_to_token(i) for i in inputs), flush=True)
        print(' '.join(self.tokenizer.id_to_token(o) if o >= 0 else "-1" for o in outputs), flush=True)

class DatasetShift(Dataset):
    def __init__(self, file, offset: int, n_gpus: int, tokenizer, seq_length=512, mask_p=0.15, short_p=0.1, random_p=0.1, keep_p=0.1):
        self.tokenizer = tokenizer

        self.seq_length = seq_length
        self.short_p = short_p
        self.n_special_tokens = 6

        self.masking_strategy = SpanMaskingStrategy(mask_p, tokenizer, self.n_special_tokens, padding_label_id=-100, random_p=random_p, keep_p=keep_p)

        self.mask_index = self.tokenizer.token_to_id("[MASK]")
        self.cls_index = self.tokenizer.token_to_id("[CLS]")
        self.sep_index = self.tokenizer.token_to_id("[SEP]")
        self.pad_index = self.tokenizer.token_to_id("[PAD]")

        self.segments = []
        for i, segment in enumerate(open(file)):
            if i % n_gpus != offset:
                continue

            segment = segment.strip().split(" ")
            assert len(segment) <= seq_length - 2, " ".join(segment)
            segment = [self.tokenizer.token_to_id(token) for token in segment]
            self.segments.append(segment)

    def __len__(self):
        return len(self.segments)

    def rand(self):
        return torch.rand(1).item()

    def randint(self, low, high):
        return torch.randint(low=low, high=high, size=(1,)).item()

    def __getitem__(self, index):
        tokens = self.segments[index]

        target_seq_length = self.seq_length - 2 if self.rand() > self.short_p else self.randint(1, self.seq_length - 2)
        tokens = tokens[:target_seq_length]
        padding_length = (self.seq_length - 2) - len(tokens)
        segment = [self.cls_index] + tokens + [self.sep_index] + [self.pad_index] * padding_length
        segment = torch.LongTensor(segment)
        segment2 = [self.cls_index] + random.sample(tokens, len(tokens)) + [self.sep_index] + [self.pad_index] * padding_length
        segment2 = torch.LongTensor(segment2)

        attention_mask = torch.cat([
            torch.zeros(len(tokens) + 2, dtype=torch.bool),
            torch.ones(padding_length, dtype=torch.bool)
        ])

        inputs, outputs, seg_mask = self.masking_strategy(segment)

        seg_mask[len(tokens)+2:] = torch.ones(padding_length, dtype=torch.bool)

        return inputs[:-1], attention_mask[:-1], outputs[1:]

    def show_random_item(self):
        inputs, _, outputs = self.__getitem__(self.randint(0, len(self)))
        print(' '.join(self.tokenizer.id_to_token(i) for i in inputs), flush=True)
        print(' '.join(self.tokenizer.id_to_token(o) if o >= 0 else "-1" for o in outputs), flush=True)

class DatasetMCLM(Dataset):
    def __init__(self, file, offset: int, n_gpus: int, tokenizer, seq_length=512, mask_p=0.15, short_p=0.1, random_p=0.1, keep_p=0.1):
        self.tokenizer = tokenizer

        self.seq_length = seq_length
        self.short_p = short_p
        self.n_special_tokens = 6
        self.padding_label_id=-100
        self.mlm_ratio = 1.0 / (1.0 + mask_p)

        self.masking_strategy = SpanMaskingStrategy(mask_p, tokenizer, self.n_special_tokens, padding_label_id=-100, random_p=random_p, keep_p=keep_p)

        self.mask_index = self.tokenizer.token_to_id("[MASK]")
        self.cls_index = self.tokenizer.token_to_id("[CLS]")
        self.sep_index = self.tokenizer.token_to_id("[SEP]")
        self.pad_index = self.tokenizer.token_to_id("[PAD]")

        self.segments = []
        for i, segment in enumerate(open(file)):
            if i % n_gpus != offset:
                continue

            segment = segment.strip().split(" ")
            assert len(segment) <= seq_length - 2, " ".join(segment)
            segment = [self.tokenizer.token_to_id(token) for token in segment]
            self.segments.append(segment)

    def __len__(self):
        return len(self.segments)

    def rand(self):
        return torch.rand(1).item()

    def randint(self, low, high):
        return torch.randint(low=low, high=high, size=(1,)).item()

    def __getitem__(self, index):
        tokens = self.segments[index]

        target_seq_length = self.seq_length - 2 if self.rand() > self.short_p else self.randint(1, self.seq_length - 2)
        tokens = tokens[:target_seq_length]
        padding_length = (self.seq_length - 1) - len(tokens) + 1
        segment = [self.cls_index] + tokens + [self.pad_index] * padding_length
        segment = torch.LongTensor(segment)
        causal_outputs = tokens + [self.padding_label_id] * padding_length
        causal_outputs = torch.LongTensor(causal_outputs)

        attention_mask = torch.cat([
            torch.zeros(len(tokens) + 1, dtype=torch.bool),
            torch.ones(padding_length, dtype=torch.bool)
            ])[:-1].unsqueeze(0).expand(self.seq_length, -1)

        causal_attention_mask = torch.ones(self.seq_length, self.seq_length, dtype=torch.bool).triu(diagonal=1)

        if self.rand() < self.mlm_ratio:
            inputs, outputs, seg_mask = self.masking_strategy(segment)
            inputs = inputs[:-1]
            outputs = outputs[1:]
            causal = torch.zeros(1, dtype=torch.bool)
        else:
            inputs = segment[:-1]
            attention_mask = causal_attention_mask
            outputs = causal_outputs
            causal = torch.ones(1, dtype=torch.bool)

        return inputs, attention_mask, outputs, causal

    def show_random_item(self):
        inputs, _, outputs = self.__getitem__(self.randint(0, len(self)))
        print(' '.join(self.tokenizer.id_to_token(i) for i in inputs), flush=True)
        print(' '.join(self.tokenizer.id_to_token(o) if o >= 0 else "-1" for o in outputs), flush=True)

class DatasetMLM(Dataset):
    def __init__(self, file, offset: int, n_gpus: int, tokenizer, seq_length=512, mask_p=0.15, short_p=0.1, random_p=0.1, keep_p=0.1):
        self.tokenizer = tokenizer

        self.seq_length = seq_length
        self.short_p = short_p
        self.n_special_tokens = 6

        self.masking_strategy = SpanMaskingStrategy(mask_p, tokenizer, self.n_special_tokens, padding_label_id=-100, random_p=random_p, keep_p=keep_p)

        self.mask_index = self.tokenizer.token_to_id("[MASK]")
        self.cls_index = self.tokenizer.token_to_id("[CLS]")
        self.pad_index = self.tokenizer.token_to_id("[PAD]")

        self.segments = []
        for i, segment in enumerate(open(file)):
            if i % n_gpus != offset:
                continue

            segment = segment.strip().split(" ")
            assert len(segment) <= seq_length - 2, " ".join(segment)
            segment = [self.tokenizer.token_to_id(token) for token in segment]
            self.segments.append(segment)

    def __len__(self):
        return len(self.segments)

    def rand(self):
        return torch.rand(1).item()

    def randint(self, low, high):
        return torch.randint(low=low, high=high, size=(1,)).item()

    def __getitem__(self, index):
        tokens = self.segments[index]

        target_seq_length = self.seq_length - 2 if self.rand() > self.short_p else self.randint(1, self.seq_length - 2)
        tokens = tokens[:target_seq_length]
        padding_length = (self.seq_length - 1) - len(tokens) + 1
        segment = [self.cls_index] + tokens + [self.pad_index] * padding_length
        segment = torch.LongTensor(segment)

        attention_mask = torch.cat([
            torch.zeros(len(tokens) + 1, dtype=torch.bool),
            torch.ones(padding_length, dtype=torch.bool)
        ])[:-1].unsqueeze(0).expand(self.seq_length, -1)

        inputs, outputs, _ = self.masking_strategy(segment)

        return inputs[:-1], attention_mask, outputs[1:]

    def show_random_item(self):
        inputs, _, outputs = self.__getitem__(self.randint(0, len(self)))
        print(' '.join(self.tokenizer.id_to_token(i) for i in inputs), flush=True)
        print(' '.join(self.tokenizer.id_to_token(o) if o >= 0 else "-1" for o in outputs), flush=True)


class DatasetCausal(Dataset):
    def __init__(self, file, offset: int, n_gpus: int, tokenizer, seq_length=512):
        self.tokenizer = tokenizer

        self.seq_length = seq_length
        self.n_special_tokens = 6
        self.padding_label_id = -100

        self.mask_index = self.tokenizer.token_to_id("[MASK]")
        self.cls_index = self.tokenizer.token_to_id("[CLS]")
        self.pad_index = self.tokenizer.token_to_id("[PAD]")

        self.segments = []
        for i, segment in enumerate(open(file)):
            if i % n_gpus != offset:
                continue

            segment = segment.strip().split(" ")
            assert len(segment) <= seq_length - 2, " ".join(segment)
            segment = [self.tokenizer.token_to_id(token) for token in segment]
            self.segments.append(segment)

    def __len__(self):
        return len(self.segments)

    def rand(self):
        return torch.rand(1).item()

    def randint(self, low, high):
        return torch.randint(low=low, high=high, size=(1,)).item()

    def __getitem__(self, index):
        tokens = self.segments[index]

        target_seq_length = self.seq_length - 2
        tokens = tokens[:target_seq_length]
        padding_length = (self.seq_length - 1) - len(tokens) + 1
        segment = [self.cls_index] + tokens + [self.pad_index] * padding_length
        segment = torch.LongTensor(segment)

        attention_mask = torch.ones(self.seq_length, self.seq_length, dtype=torch.bool).triu(diagonal=1)

        inputs = segment[:-1]
        outputs = tokens + [self.padding_label_id] * padding_length
        outputs = torch.LongTensor(outputs)

        return inputs, attention_mask, outputs

    def show_random_item(self):
        inputs, _, outputs = self.__getitem__(self.randint(0, len(self)))
        print(' '.join(self.tokenizer.id_to_token(i) for i in inputs), flush=True)
        print(' '.join(self.tokenizer.id_to_token(o) if o >= 0 else "-1" for o in outputs), flush=True)
