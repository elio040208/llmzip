# llmzip Benchmarks

Run:

```bash
.venv/bin/python test/benchmark.py
```

If no input file is provided, the script downloads Project Gutenberg's Pride and
Prejudice, removes the boilerplate markers, normalizes whitespace into a
single-space text stream, and keeps the first 2 KiB by default before comparing:

- `llmzip`
- `gzip -9`
- `bzip2 -9`
- `zstd -19` if installed
- `brotli -q 11` if installed
- `xz -9` if installed

Outputs:

- downloaded/input text under `test/data/`
- compressed artifacts under `test/out/`
- CSV summary at `test/benchmark_results.csv`

`llmzip_file` includes the compact binary header and the arithmetic-coded
payload. The row also reports `payload_bpb` to separate the language model
payload from archive metadata.

The default benchmark is intentionally small and whitespace-normalized:

- raw Gutenberg text has many blank lines, indentation, and line wraps
- the current SentencePiece tokenizer is not byte-exact for whitespace
- llmzip now rejects inputs that do not round-trip byte-exactly through the
  tokenizer
- a raw whitespace-heavy file would mostly test tokenizer normalization failure,
  not the LLM probability model

The current implementation uses KV cache, but the Python arithmetic coder and
full-vocabulary frequency conversion are still slow. The default generated file
is 2 KiB so benchmark runs stay quick.
