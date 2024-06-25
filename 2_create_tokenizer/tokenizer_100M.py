import argparse
import json
from collections import Counter
from tqdm import tqdm
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer, pre_tokenizers, decoders, normalizers, Regex, processors
import tokenization_scorer


def initialize_tokenizer(args):
    start_of_text_symbol = chr(0x2402)  # ␂
    end_of_text_symbol = chr(0x2403)  # ␃
    unk_symbol = chr(0x2426)  # ␦
    mask_symbol = chr(0x2425)  # ␥
    pad_symbol = chr(0x2422)  # ␢
    newline_symbol = chr(0x2424)  # ␤
    whitespace_symbol = chr(0x2423)  # ␣

    special_tokens = [unk_symbol, start_of_text_symbol, end_of_text_symbol, pad_symbol, mask_symbol]
    special_tokens += [chr(0x241A) + f"_{i}" for i in range(11)]
    special_tokens += [f"<0x{i:02X}>" for i in range(256)]

    tokenizer = Tokenizer(BPE(
        unk_token=unk_symbol,
        byte_fallback=True,
        fuse_unk=True
    ))

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Prepend(" "),
        normalizers.NFKC(),
        normalizers.Replace(Regex("\n"), '\n '),
        normalizers.Replace(Regex(" *\n"), '\n'),
        normalizers.Replace(Regex("\t"), "    "),
        normalizers.Replace(Regex("\n"), newline_symbol),
        normalizers.Replace(Regex("\s"), whitespace_symbol)
    ])

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([

        pre_tokenizers.Split(
            Regex("\u2424{1}"),
            behavior="isolated",
            invert=False
        ),

        # do standard whitespace splitting
        pre_tokenizers.Split(
            Regex("\u2423{1}[^\u2423\u2424]+"),
            behavior="isolated",
            invert=False
        ),

        # split digits and non-Latin-alnums
        pre_tokenizers.Split(
            Regex("\u2423{0,1}\d{1}"),
            behavior="isolated",
            invert=False
        ),
        pre_tokenizers.Split(
            Regex("\u2423{0,1}[^\\s\u2423\u2424A-Za-z0-9\u00C0-\u00FF\u0100-\u017F\u1E00-\u1EFF]{1}"),
            behavior="isolated",
            invert=False
        ),

        # allow at most 8 spaces
        pre_tokenizers.Split(
            Regex("\u2423{2,8}"),
            behavior="isolated",
            invert=False
        )
    ])

    tokenizer.decoder = decoders.Sequence([
        decoders.ByteFallback(),
        decoders.Fuse(),
        decoders.Strip(whitespace_symbol, 1, 0),
        decoders.Replace(newline_symbol, '\n'),
        decoders.Replace(whitespace_symbol, ' '),
        decoders.Replace('\n ', "\n")
    ])

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{start_of_text_symbol} $A {end_of_text_symbol}",
        pair=f"{start_of_text_symbol} $A {end_of_text_symbol} $B {end_of_text_symbol}",
        special_tokens=[
            (start_of_text_symbol, 1),
            (end_of_text_symbol, 2)
        ]
    )

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=args.min_frequency,
        show_progress=True
    )

    return tokenizer, trainer


def calculate_stats(tokenizer, args):
    counter, n_words = Counter(), 0
    all_tokens = []
    for i, document in enumerate(open(f"{args.validation_path}")):
        text = json.loads(document).strip()
        if len(text) > 0:
            n_words += len(text.split())
            encoding = tokenizer.encode(text)
            tokens = encoding.tokens
            counter.update(tokens)
            all_tokens += tokens

            if i == 0:
                print("Example of tokenization:")
                print(text)
                print(tokenizer.decode(encoding.ids))
                for j in encoding.ids:
                    print(j, tokenizer.id_to_token(j))

    sorted_subwords = counter.most_common()

    n_subwords = sum(freq for _, freq in sorted_subwords)
    print(f"Average splits per word: {n_subwords / n_words:.3f}", flush=True)

    f_95 = sorted_subwords[len(sorted_subwords) * 95 // 100][1]
    renyi_score = tokenization_scorer.score(' '.join(all_tokens), metric="renyi", power=2.5)

    print(f"F_{{95%}} is {f_95}\n")
    print(f"Renyi score is {renyi_score}\n")

    with open(f"{args.vocab_path[:-5]}_stats.txt", "w") as f:
        f.write(f"Vocabulary size: {args.vocab_size}\n")
        f.write(f"Average splits per word: {n_subwords / n_words:.3f}\n")
        f.write(f"F_{{95%}} is {f_95}\n")
        f.write(f"Renyi score is {renyi_score}\n\n")
        sorted_subwords_str = '\n\t'.join(f"{freq}: {subword}" for subword, freq in sorted_subwords)
        f.write(f"Sorted subwords:\n\t{sorted_subwords_str}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT sharding')
    parser.add_argument('--input_path', type=str, default="../data/train_100M.jsonl", help='Specify the input filename')
    parser.add_argument('--validation_path', type=str, default="../data/dev.jsonl", help='Specify the validation filename')
    parser.add_argument('--vocab_path', type=str, default="../tokenizer_100M.json", help='Specify the output filename')
    parser.add_argument('--vocab_size', type=int, default=2**14, help='Number of subwords in the trained tokenizer')
    parser.add_argument('--min_frequency', type=int, default=10, help='Minimal number of occurences of every candidate subword')
    args = parser.parse_args()

    print(f"Initializing a BPE tokenizer", flush=True)
    tokenizer, trainer = initialize_tokenizer(args)

    print("Training the tokenizer", flush=True)
    def iterator(file_path: str):
        for line in tqdm(open(file_path)):
            text = json.loads(line).strip()
            if len(line) == 0:
                continue
            yield line

    tokenizer.train_from_iterator(iterator(args.input_path), trainer)

    print("Saving the tokenizer", flush=True)
    tokenizer.save(args.vocab_path)

    with open(args.vocab_path) as f:
        tokenizer_json = json.load(f)
    tokenizer_json["added_tokens"] = tokenizer_json["added_tokens"][:-256]
    with open(args.vocab_path, "w") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False, indent=4)

    print("TEST")
    print("Trying to load the tokenizer...")
    tokenizer = Tokenizer.from_file(args.vocab_path)
    print("Success!")

    calculate_stats(tokenizer, args)

    print("Samples from the tokenizer:")

    def test(tokenizer, text):
        subwords = tokenizer.encode(text).tokens
        return ' '.join(subwords)

    texts = [
        """One of the most impressive long term hobby projects is Robert's Rocket Project. He started building a 100 lbf liquid engine in 2001, fired a regeneratively cooled version in 2007, started building a regen 250 lbf in 2008.""",
        """what are examples of interfaces that allow you to manage sets of queries (SQL, splunk, lucene/elastic, xpath, whatever other language)?""",
        """### Increasingly seeing a big schism between what I think my research is & what others think it is. I don't do qualitative work and I'm not trained in anthro or theories of race or gender. I can't supervise students with these interests! I'm a sociophonetician who works on prosody!""",
        """The Northern Lights season is here... Taking these pictures is an art itself and requires preparation, so The Local spoke to an expert to find out how to take awe-inspiring snaps of the Northern Lights.""",
        """Some people have SOTA facial recognition abilities: "At the very upper end of the performance scale, a cohort of just 1-2% of the population are 'super-recognisers'-people who can memorise and recall unfamiliar faces, even after the briefest glimpse.\""""
    ]

    for text in texts:
        print(f"INPUT:  {text}\nTOKENS: {test(tokenizer, text)}\n", flush=True)
