from __future__ import annotations

import argparse
import json

from llmzip import compress_file, decompress_file


DEFAULT_CHECKPOINT = "./models/baseline_sp1024.pt"
DEFAULT_TOKENIZER = "./tokenizers/fineweb_1024_bpe.model"


def main() -> None:
    parser = argparse.ArgumentParser(prog="llmzip")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    common.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)

    p_compress = sub.add_parser("compress", parents=[common])
    p_compress.add_argument("input")
    p_compress.add_argument("output")
    p_compress.add_argument("--total-freq", type=int, default=1 << 20)
    p_compress.add_argument("--max-context", type=int, default=1024)

    p_decompress = sub.add_parser("decompress", parents=[common])
    p_decompress.add_argument("input")
    p_decompress.add_argument("output")

    args = parser.parse_args()
    if args.cmd == "compress":
        result = compress_file(
            args.input,
            args.output,
            checkpoint_path=args.checkpoint,
            tokenizer_path=args.tokenizer,
            total_freq=args.total_freq,
            max_context=args.max_context,
        )
    else:
        result = decompress_file(
            args.input,
            args.output,
            checkpoint_path=args.checkpoint,
            tokenizer_path=args.tokenizer,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
