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

`llmzip_file` includes the JSON header and byte-sidecar patch. The row also
reports `payload_bpb` and `payload_plus_sidecar_bpb` to separate the language
model arithmetic-coded payload from archive metadata.

The default benchmark is intentionally small and whitespace-normalized:

- raw Gutenberg text has many blank lines, indentation, and line wraps
- the current SentencePiece tokenizer is not byte-exact for whitespace
- those bytes must be restored through the sidecar patch
- a raw whitespace-heavy file would mostly measure sidecar overhead, not the
  LLM probability model

For the current no-KV-cache implementation, the default generated file is 2 KiB.
Larger files work, but `llmzip` recomputes the full context for every token and
will be slow.
