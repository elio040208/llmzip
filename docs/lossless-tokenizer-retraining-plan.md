# Lossless Tokenizer and Retraining Plan

## Problem

The current llmzip model uses Parameter Golf's `sp1024` SentencePiece tokenizer:

```text
tokenizers/fineweb_1024_bpe.model
```

This tokenizer is not byte-exact for all UTF-8 text. It can normalize or drop
some whitespace patterns, including:

- repeated spaces
- leading spaces
- trailing newlines

That means:

```text
tokenizer.decode(tokenizer.encode(text)) != text
```

For a lossless compressor, this is unacceptable unless the archive stores an
extra byte-sidecar patch. We intentionally removed sidecar from `LLMZ3`, so the
compressor now rejects non-round-tripping inputs.

The permanent fix is:

```text
Train and use a tokenizer/model pair whose tokenization is byte-exact.
```

## Recommended Direction

Use a pure byte-level tokenizer.

Parameter Golf already contains a pure-byte tokenizer concept in:

```text
/Users/cyc/Desktop/parameter-golf/data/download_hf_docs_and_tokenize.py
```

The relevant class is:

```python
class PureByteTokenizer:
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3
    byte_offset: int = 4
    byte_count: int = 256
```

So the vocab is:

```text
0   PAD
1   BOS
2   EOS
3   UNK
4   byte 0x00
5   byte 0x01
...
259 byte 0xff
```

Total vocab:

```text
260 tokens
```

This is exactly lossless because every byte maps to one token:

```text
token_id = byte_value + 4
byte_value = token_id - 4
```

Repeated spaces, leading spaces, tabs, newlines, emoji, CJK text, and arbitrary
UTF-8 bytes all survive because the tokenizer does not normalize text at all.

## Why Not Just Fix SentencePiece?

There is a possible SentencePiece route:

```text
normalization_rule_name = identity
remove_extra_whitespaces = False
add_dummy_prefix = False
byte_fallback = True
```

This can improve whitespace round-tripping for UTF-8 text.

But for a compressor, SentencePiece is still less clean than byte-level because:

- you must audit every normalization option
- you must test every whitespace/control-character case
- SentencePiece decode behavior is still a dependency
- arbitrary bytes are not naturally represented unless you stay in UTF-8 text land

For a permanent lossless compression project, pure byte tokens are the most
defensible baseline.

## Tradeoff

Byte-level tokenization is lossless but usually gives longer token sequences.

Example:

```text
"The quick brown fox"
```

SentencePiece might tokenize this into a few subword tokens:

```text
["The", " quick", " brown", " fox"]
```

Byte-level tokenization turns it into one token per byte:

```text
T h e   q u i c k   b r o w n   f o x
```

That means:

- more autoregressive steps
- slower compression/decompression
- more pressure on KV cache
- the model must learn byte-level spelling and whitespace patterns

But for llmzip, correctness comes first:

```text
No byte-sidecar, no hidden patch, no tokenizer normalization failure.
```

## Plan Overview

Recommended plan:

```text
Phase 1: Build/download byte260 dataset
Phase 2: Patch Parameter Golf train_gpt.py to support pure-byte tokenizer
Phase 3: Train a byte-level GPT checkpoint
Phase 4: Add byte tokenizer support to llmzip
Phase 5: Benchmark sp1024 vs byte260 fairly
```

## Phase 1: Build or Download byte260 Data

Parameter Golf's downloader supports tokenizer variants:

```text
sp1024
sp4096
byte260
```

The data docs mention:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

For byte-level training, try:

```bash
cd /Users/cyc/Desktop/parameter-golf
.venv/bin/python data/cached_challenge_fineweb.py --variant byte260
```

If you want more training shards:

```bash
.venv/bin/python data/cached_challenge_fineweb.py \
  --variant byte260 \
  --train-shards 80
```

Expected dataset path:

```text
/Users/cyc/Desktop/parameter-golf/data/datasets/fineweb10B_byte260/
```

Expected tokenizer artifact:

```text
/Users/cyc/Desktop/parameter-golf/data/tokenizers/fineweb_pure_byte_260.json
```

If the published byte260 export is not available locally, use the retokenizer
workflow:

```bash
cd /Users/cyc/Desktop/parameter-golf
.venv/bin/python data/download_hf_docs_and_tokenize.py \
  --output-root /Users/cyc/Desktop/parameter-golf/data/byte260_export \
  --tokenizer-config ./data/tokenizer_specs.json
```

But `tokenizer_specs.json` currently only lists `sp_bpe_1024`. Add a byte spec:

```json
{
  "name": "pure_byte_260",
  "dataset_suffix": "byte260",
  "kind": "byte",
  "filename": "fineweb_pure_byte_260.json"
}
```

The script already has `build_pure_byte_tokenizer()`, so this route is supported
by the data tooling conceptually.

## Phase 2: Patch train_gpt.py for Pure Byte

The current root `train_gpt.py` assumes SentencePiece:

```python
if not args.tokenizer_path.endswith(".model"):
    raise ValueError(...)
sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
```

This must be generalized.

Add an env/config option:

```text
TOKENIZER_KIND=sentencepiece | byte
```

For byte training:

```text
TOKENIZER_KIND=byte
VOCAB_SIZE=260
TOKENIZER_PATH=./data/tokenizers/fineweb_pure_byte_260.json
DATA_PATH=./data/datasets/fineweb10B_byte260
```

The model architecture can stay mostly the same, but the embedding/head vocab
size changes from:

```text
1024 -> 260
```

That reduces embedding and output-head parameters, but the sequence length in
bytes becomes longer than subword token length.

### Validation BPB for Byte Tokens

Current validation BPB uses SentencePiece byte-length lookup tables:

```python
base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(...)
```

For byte tokens, BPB accounting is simpler:

```text
token id 0 PAD -> 0 bytes
token id 1 BOS -> 0 bytes
token id 2 EOS -> 0 bytes
token id 3 UNK -> probably invalid / 0 bytes
token id >= 4  -> 1 byte
```

So the byte-count LUT is:

```python
byte_lut = torch.zeros(vocab_size, dtype=torch.int16, device=device)
byte_lut[4:260] = 1
```

Then validation BPB is:

```text
loss_bits_per_token * token_count / byte_count
```

Because byte tokens map directly to bytes, this becomes especially clean.

## Phase 3: Train a Byte-Level GPT

Start with a short run to verify the pipeline:

```bash
cd /Users/cyc/Desktop/parameter-golf

TOKENIZER_KIND=byte \
DATA_PATH=./data/datasets/fineweb10B_byte260 \
TOKENIZER_PATH=./data/tokenizers/fineweb_pure_byte_260.json \
VOCAB_SIZE=260 \
ITERATIONS=100 \
MAX_WALLCLOCK_SECONDS=600 \
RUN_ID=byte260_smoke \
.venv/bin/torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Then scale to your hardware.

For 4x RTX 3090, likely command shape:

```bash
TOKENIZER_KIND=byte \
DATA_PATH=./data/datasets/fineweb10B_byte260 \
TOKENIZER_PATH=./data/tokenizers/fineweb_pure_byte_260.json \
VOCAB_SIZE=260 \
TRAIN_SEQ_LEN=1024 \
TRAIN_BATCH_TOKENS=524288 \
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=21600 \
RUN_ID=byte260_4x3090 \
.venv/bin/torchrun --standalone --nproc_per_node=4 train_gpt.py
```

Notes:

- 4x3090 should be enough to run the training pipeline if the existing script
  already runs on your environment.
- Byte-level tokens increase sequence count per byte, so for compression quality
  you may eventually want longer context, for example `TRAIN_SEQ_LEN=2048`.
- Longer context increases memory use. Start with 1024, verify, then tune.

## Phase 4: Add Byte Tokenizer Support to llmzip

llmzip currently loads SentencePiece:

```python
spm.SentencePieceProcessor(model_file=str(tokenizer_path))
```

For byte260, add a tokenizer abstraction:

```python
class Tokenizer:
    vocab_size: int
    bos_id: int
    def encode_bytes(data: bytes) -> list[int]: ...
    def decode_bytes(tokens: list[int]) -> bytes: ...
```

SentencePiece implementation:

```text
bytes -> UTF-8 text -> sp.encode -> tokens
tokens -> sp.decode -> UTF-8 bytes
```

Byte implementation:

```python
def encode_bytes(data):
    return [byte + 4 for byte in data]

def decode_bytes(tokens):
    return bytes(token - 4 for token in tokens)
```

Then llmzip no longer needs this rejection for byte mode:

```python
if normalized_data != data:
    raise ValueError(...)
```

For byte mode, roundtrip is guaranteed.

## Phase 5: Model Shape Compatibility

The current llmzip `GPT()` constructor defaults to:

```text
vocab_size = 1024
num_layers = 9
model_dim = 512
num_heads = 8
num_kv_heads = 4
```

If the byte model is trained with:

```text
VOCAB_SIZE=260
```

then llmzip must load the model with:

```python
GPT(vocab_size=260)
```

The loader currently hardcodes:

```python
model = GPT()
```

So add either:

```text
--vocab-size 260
```

or infer vocab size from checkpoint:

```python
vocab_size = state["tok_emb.weight"].shape[0]
model = GPT(vocab_size=vocab_size)
```

Inferring from checkpoint is better because it avoids mismatched CLI settings.

## Phase 6: Benchmark

After byte model training and llmzip byte tokenizer support, benchmark these:

```text
1. sp1024 model on whitespace-normalized text
2. byte260 model on raw text
3. gzip / zstd / brotli on raw text
```

Important:

```text
Do not normalize the byte260 benchmark input.
```

The whole point is that byte260 can compress raw bytes losslessly without
sidecar.

Metrics to compare:

```text
payload_bpb
bpb_file
archive_bytes
compress_seconds
decompress_seconds
```

Expected behavior:

- byte260 will be universally lossless
- byte260 will likely be slower because it has more tokens
- byte260 may need more training or longer context to beat strong classic compressors
- but its `bpb_file` will be honest on raw text

## Minimal Patch Checklist

Parameter Golf:

```text
[ ] Add TOKENIZER_KIND env var
[ ] Support pure-byte tokenizer path
[ ] Allow non-.model tokenizer when TOKENIZER_KIND=byte
[ ] Set VOCAB_SIZE=260
[ ] Build byte BPB LUT
[ ] Verify byte260 train/val shards load
[ ] Run 100-iteration smoke training
[ ] Run longer 4x3090 training
```

llmzip:

```text
[ ] Add tokenizer abstraction
[ ] Add byte tokenizer implementation
[ ] Infer GPT vocab_size from checkpoint
[ ] Support tokenizer_kind in web/API
[ ] Remove SentencePiece-only UTF-8 restriction for byte mode
[ ] Benchmark raw text with byte260
```

## Final Recommendation

For a general-purpose LLM compressor, use byte-level tokenization.

SentencePiece is useful for language modeling efficiency, but for a lossless
compressor it creates a conceptual problem:

```text
The tokenizer is allowed to change the byte sequence.
```

Byte-level tokenization removes that entire class of bugs:

```text
The model predicts bytes.
The arithmetic coder encodes bytes.
The decoder restores bytes.
```

That is the clean long-term architecture.
