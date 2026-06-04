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
python main.py compress input.txt output.llmz
python main.py decompress output.llmz restored.txt
```

The archive stores a JSON header with:

- original token count
- original byte count
- original SHA256
- model SHA256
- tokenizer SHA256
- arithmetic coding frequency precision
- max context length

The decoder verifies the model/tokenizer hashes and the restored text hash.

## Current Limits

- UTF-8 text only.
- The Parameter Golf `sp1024` tokenizer normalizes some whitespace, including
  trailing newlines, leading spaces, and repeated spaces. `llmzip` stores a
  compressed byte-sidecar patch so decoded archives still restore the original
  bytes exactly.
- Full-vocabulary arithmetic coding.
- No KV cache yet, so long files are slow.
- The model is assumed to be pre-shared and is not included in `.llmz`.
