# llmzip

Lossless UTF-8 text compression with a local language model and arithmetic coding.

This first version is intentionally small and explicit. It is wired for the
Parameter Golf `baseline_sp1024.pt` checkpoint and the matching SentencePiece
tokenizer:

```text
/Users/cyc/Desktop/parameter-golf/baseline_sp1024.pt
/Users/cyc/Desktop/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
```

## Usage

```bash
python main.py
```

Then open `http://127.0.0.1:8765`.

The web UI lets you enter text, compress it, inspect the compact `.llmz` archive
as hex/base64/header metadata, and decompress the base64 archive back to text.

The archive stores a compact binary header with:

- original token count
- original byte count
- payload byte count
- arithmetic coding frequency precision
- max context length
- BOS token id
- logit mode
- CRC32

The decoder verifies restored bytes with CRC32. The model and tokenizer are
assumed to be pre-shared by the encoder and decoder.

## Current Limits

- UTF-8 text only.
- The Parameter Golf `sp1024` tokenizer normalizes some whitespace, including
  trailing newlines, leading spaces, and repeated spaces. `llmzip` rejects
  inputs that do not round-trip byte-exactly through this tokenizer. Normalize
  whitespace first or switch to a byte-level tokenizer.
- Full-vocabulary arithmetic coding.
- KV cache is used during token-by-token coding. The implementation rebuilds
  the cache when the sliding context window advances so RoPE positions match
  the original full-prefix behavior.
- The model is assumed to be pre-shared and is not included in `.llmz`.
