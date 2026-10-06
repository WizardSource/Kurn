"""WikiText-2 (raw, test split) -> token ids for `engine MODEL ppl T CTX tokens.txt`.

    python tokenize_wikitext.py test-00000-of-00001.parquet tokenizer.json tokens.txt

The parquet file is Salesforce/wikitext wikitext-2-raw-v1; texts are joined with "\\n\\n" as in
llama.cpp's perplexity setup. tokenizer.json is the model's Hugging Face tokenizer.
"""

import sys

import pyarrow.parquet as pq
from tokenizers import Tokenizer


def main(parquet, tok_json, out):
    text = "\n\n".join(pq.read_table(parquet).column("text").to_pylist())
    ids = Tokenizer.from_file(tok_json).encode(text, add_special_tokens=False).ids
    with open(out, "w") as fh:
        fh.write(" ".join(map(str, ids)) + "\n")
    print(f"{len(ids)} tokens -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
