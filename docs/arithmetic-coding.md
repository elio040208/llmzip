# Arithmetic Coding in llmzip

This note explains how llmzip turns LLM predictions into a compressed payload,
and how the decoder reconstructs the original token sequence from that payload.

The relevant code lives in:

- `llmzip/codec.py`: connects tokenizer, model logits, frequency tables, and the archive format
- `llmzip/arithmetic.py`: bit-level arithmetic encoder and decoder

## The Big Picture

llmzip does not store the model's probability table in the compressed file.
Instead, both encoder and decoder run the same model.

The encoder does this:

```text
prefix -> LLM -> next-token probabilities -> arithmetic encode real token
```

The decoder does this:

```text
same prefix -> same LLM -> same probabilities -> arithmetic decode token
```

After one token is encoded or decoded, both sides append that token to the
prefix. Therefore the next model call sees the same context on both sides.

That is the central trick:

```text
The compressed file stores the information needed to choose among model
predictions, not the predictions themselves.
```

## Concrete Example

Assume the original text is:

```text
The quick brown fox
```

The real project uses a 1024-token SentencePiece tokenizer. The exact token ids
depend on the tokenizer, but the flow is easier to see if we name the resulting
tokens:

```text
["The", " quick", " brown", " fox"]
```

These names are just labels for tokenizer ids. In the real code, arithmetic
coding receives integer token ids.

Before the first text token, both sides start from the same BOS token:

```text
prefix = [BOS]
```

The model predicts the next token. Suppose it assigns these probabilities to a
few likely tokens:

```text
"The"      0.300
"A"        0.120
"In"       0.080
" quick"   0.001
... all other tokens ...
```

The true next token is `"The"`. Since `"The"` has high probability, arithmetic
coding spends few bits to encode it.

After encoding `"The"`, the prefix becomes:

```text
[BOS, "The"]
```

Now the model predicts the next token. Suppose it says:

```text
" quick"   0.180
" first"   0.050
" same"    0.020
"The"      0.001
... all other tokens ...
```

The true token is `" quick"`. Again, because the model expects it reasonably
well, arithmetic coding spends relatively few bits.

After several steps:

```text
BOS -> "The" -> " quick" -> " brown" -> " fox"
```

The arithmetic payload is essentially a very compact set of choices made inside
the model's predicted distributions.

## Why Logits Need Integer Frequencies

The model returns logits. llmzip converts them to probabilities with softmax in
`logits_to_cumulative()`:

```python
probs = torch.softmax(logits.float().cpu(), dim=-1).numpy().astype(np.float64)
```

Arithmetic coding in this implementation does not consume floating-point
probabilities directly. It consumes integer cumulative frequencies.

The default total frequency is:

```python
DEFAULT_TOTAL_FREQ = 1 << 20
```

So every next-token probability distribution is quantized into:

```text
1,048,576 integer frequency units
```

For example, suppose the model distribution at one step is simplified as:

```text
"The"       probability 0.300
"A"         probability 0.120
"In"        probability 0.080
other ids   probability 0.500
```

With a smaller example `total_freq = 1000`, this becomes approximately:

```text
"The"       freq 300
"A"         freq 120
"In"        freq 80
other ids   freq 500
```

The real code uses `1 << 20`, not 1000, so the quantization is much finer.

The implementation also guarantees every token gets at least frequency 1:

```python
scaled = probs * (total_freq - vocab_size)
floors = np.floor(scaled).astype(np.int64)
freqs = floors + 1
```

This matters because a frequency of zero would mean "impossible to encode".
Even if the model thinks a token is extremely unlikely, the compressor must
still be able to encode it if it appears in the real text.

## Cumulative Frequencies

Arithmetic coding needs cumulative intervals. The helper is:

```python
def frequencies_to_cumulative(freqs):
    total = 0
    cumulative = [0]
    for freq in freqs:
        total += int(freq)
        cumulative.append(total)
    return cumulative
```

If the simplified frequency table is:

```text
token id for "The"       300
token id for "A"         120
token id for "In"        80
all later ids combined   500
```

Then the cumulative table divides `[0, 1000)` like this:

```text
"The"       [0, 300)
"A"         [300, 420)
"In"        [420, 500)
other ids   [500, 1000)
```

In the real code, every one of the 1024 token ids has its own interval.

## Arithmetic Coding as Interval Selection

The encoder keeps an integer interval:

```python
self.low = 0
self.high = MAX_RANGE
```

The constants are:

```python
STATE_BITS = 32
FULL_RANGE = 1 << STATE_BITS
MAX_RANGE = FULL_RANGE - 1
```

So the initial interval is:

```text
[0, 2^32 - 1]
```

When encoding one token, the encoder takes the current interval and narrows it
to the sub-interval assigned to the true token:

```python
width = self.high - self.low + 1
self.high = self.low + (width * sym_high // total) - 1
self.low = self.low + (width * sym_low // total)
```

Use a human-sized interval to see the idea. Suppose the current arithmetic
interval is:

```text
[0, 999]
```

The current model distribution uses `total = 1000`, and the true token `"The"`
has cumulative interval:

```text
[0, 300)
```

Then encoding `"The"` maps the current interval to:

```text
low  = 0 + 1000 * 0   / 1000 = 0
high = 0 + 1000 * 300 / 1000 - 1 = 299
```

So after encoding `"The"`:

```text
[0, 999] -> [0, 299]
```

Now suppose the next true token is `" quick"`. Given prefix `"The"`, the model
assigns it this cumulative interval:

```text
" quick" [100, 280) out of total 1000
```

The current arithmetic interval is `[0, 299]`, whose width is 300. Encoding
`" quick"` narrows it to:

```text
low  = 0 + 300 * 100 / 1000 = 30
high = 0 + 300 * 280 / 1000 - 1 = 83
```

So:

```text
[0, 299] -> [30, 83]
```

The final compressed bitstream only needs to identify some number inside the
final interval. The narrower the interval, the more bits are needed to identify
it.

## Why High Probability Means Fewer Bits

If the model gives the real token probability 0.5, the interval keeps about
half its width.

If the model gives the real token probability 0.01, the interval keeps only
about one percent of its width.

After many tokens, the final interval width is roughly proportional to:

```text
p(token_1)
* p(token_2 | token_1)
* p(token_3 | token_1, token_2)
* ...
```

The number of bits needed is roughly:

```text
-log2(product of probabilities)
```

Which is the same as:

```text
-log2 p(token_1)
-log2 p(token_2 | token_1)
-log2 p(token_3 | token_1, token_2)
...
```

So prediction quality and compression quality are the same thing here:

```text
If the model assigns high probability to the real next tokens, the arithmetic
payload is short.
```

## Why `DEFAULT_TOTAL_FREQ` and `MAX_RANGE` Are Different

These two constants are often confused because both are integer ranges, but
they measure different things.

`MAX_RANGE` is the internal arithmetic coder state range:

```python
STATE_BITS = 32
MAX_RANGE = (1 << 32) - 1
```

It means:

```text
low/high live on a 32-bit integer number line.
```

`DEFAULT_TOTAL_FREQ` is the probability-table precision:

```python
DEFAULT_TOTAL_FREQ = 1 << 20
```

It means:

```text
each model probability distribution is quantized into 1,048,576 frequency units.
```

The frequency table is mapped into the current arithmetic interval:

```python
width = high - low + 1
new_low  = low + width * sym_low  // total_freq
new_high = low + width * sym_high // total_freq - 1
```

So:

```text
total_freq is the ruler for probabilities.
MAX_RANGE is the ruler for the arithmetic state.
```

They do not need to be equal. In fact, making them equal would be a poor choice
for this 32-bit implementation. The frequency total should be much smaller than
the arithmetic state range so that sub-intervals do not collapse during integer
rounding.

The current choice is:

```text
MAX_RANGE          about 4.29 billion
DEFAULT_TOTAL_FREQ about 1.05 million
```

That leaves enough room for safe interval subdivision while keeping probability
quantization fine enough for the model.

## Renormalization: Why Bits Are Emitted During Encoding

If the encoder only kept shrinking `[low, high]`, the interval would quickly
become too small for a 32-bit integer. Arithmetic coding solves this by emitting
confirmed leading bits and rescaling the interval.

The encoder checks three cases:

```python
if self.high < HALF_RANGE:
    self._write_bit_plus_pending(0)
elif self.low >= HALF_RANGE:
    self._write_bit_plus_pending(1)
    self.low -= HALF_RANGE
    self.high -= HALF_RANGE
elif self.low >= QUARTER_RANGE and self.high < THREE_QUARTER_RANGE:
    self.pending_bits += 1
    self.low -= QUARTER_RANGE
    self.high -= QUARTER_RANGE
```

### Case 1: Entire Interval Is in the Lower Half

If:

```text
high < 1/2
```

then every number in the interval starts with binary bit `0`. The encoder can
write `0` immediately.

### Case 2: Entire Interval Is in the Upper Half

If:

```text
low >= 1/2
```

then every number in the interval starts with binary bit `1`. The encoder can
write `1` immediately.

After writing `1`, the encoder subtracts the half range, because the leading
bit has already been emitted.

### Case 3: Interval Is Stuck Around the Middle

If:

```text
1/4 <= low and high < 3/4
```

the leading bit is not known yet. The interval might eventually fall to the
lower half or upper half. This is the arithmetic coding underflow case.

The code records a delayed bit:

```python
self.pending_bits += 1
```

When a later step finally decides the next real bit, all pending bits are
written as the opposite bit:

```python
def _write_bit_plus_pending(self, bit):
    self.writer.write(bit)
    inverse = 1 - bit
    while self.pending_bits:
        self.writer.write(inverse)
        self.pending_bits -= 1
```

After any of these cases, the interval is shifted left:

```python
self.low = (self.low << 1) & MAX_RANGE
self.high = ((self.high << 1) & MAX_RANGE) | 1
```

This restores precision so the next token can be encoded safely.

## BitWriter: Bits Become Bytes

The arithmetic encoder emits individual bits. The archive needs bytes.

`BitWriter.write()` shifts each bit into a byte buffer:

```python
self._current = (self._current << 1) | int(bit)
self._count += 1
if self._count == 8:
    self._bytes.append(self._current)
```

At the end, `BitWriter.finish()` pads the final partial byte with zeros:

```python
if self._count:
    self._current <<= 8 - self._count
    self._bytes.append(self._current)
```

The decoder knows the real number of tokens from `token_count`, so padding bits
do not create extra tokens.

## Decoder: Running the Process Backward

The decoder starts by reading 32 bits into `value`:

```python
self.value = 0
for _ in range(STATE_BITS):
    self.value = (self.value << 1) | self.reader.read()
```

`value` is the current 32-bit view of the compressed number.

At each token position, decoder has:

```text
low
high
value
the same cumulative table as encoder
```

It maps `value` into the cumulative-frequency coordinate:

```python
scaled = (((self.value - self.low + 1) * total) - 1) // width
symbol = bisect_right(cumulative, scaled) - 1
```

Example:

```text
current cumulative table:
"The"       [0, 300)
"A"         [300, 420)
"In"        [420, 500)
other ids   [500, 1000)
```

If `scaled = 237`, it falls inside:

```text
"The" [0, 300)
```

So the decoder outputs the token id for `"The"`.

Then the decoder performs the same interval narrowing as the encoder:

```python
self.high = self.low + (width * sym_high // total) - 1
self.low = self.low + (width * sym_low // total)
```

Then it performs the same renormalization. Whenever it shifts the interval left,
it also shifts in one more bit from the compressed payload:

```python
self.value = ((self.value << 1) & MAX_RANGE) | self.reader.read()
```

This keeps encoder and decoder synchronized.

## Why Decoder Can Reconstruct Tokens Without Storing Them

For the first token, both sides know:

```text
prefix = [BOS]
```

So both sides build the same cumulative table. The decoder uses the arithmetic
payload to choose the same first token the encoder encoded.

Then both sides have:

```text
prefix = [BOS, first_token]
```

So the second cumulative table also matches.

This repeats until `token_count` tokens have been decoded.

The archive does not need to store the text tokens directly because the
arithmetic payload plus the shared model are enough to select them.

## Where the Archive Header Fits

The compact `LLMZ2` header stores the minimum information needed to reproduce
the arithmetic process:

```text
token_count
byte_count
payload_bytes
total_freq
max_context
bos_id
logit_mode
crc32
```

The most important arithmetic fields are:

- `token_count`: how many tokens the decoder must decode
- `total_freq`: the same integer probability precision used by encoder
- `max_context`: how much prefix context the model sees
- `bos_id`: the initial token before the first real token
- `logit_mode`: whether to reproduce KV-cache or legacy full-prefix logits
- `payload_bytes`: where the arithmetic payload ends

`crc32` and `byte_count` are checks. They are not the core arithmetic mechanism,
but they help detect a wrong model, wrong tokenizer, corrupted payload, or
implementation mismatch.

## End-to-End Summary

Compression:

```text
1. Read UTF-8 bytes.
2. Tokenize to token ids.
3. Check tokenizer round-trips byte-exactly.
4. Start prefix with BOS.
5. For each real token:
   a. Run LLM on prefix.
   b. Convert logits to integer cumulative frequencies.
   c. Arithmetic-encode the real token using that table.
   d. Append the real token to prefix.
6. Finish arithmetic stream.
7. Write compact header plus payload.
```

Decompression:

```text
1. Read compact header.
2. Start prefix with BOS.
3. For token_count steps:
   a. Run LLM on prefix.
   b. Convert logits to the same integer cumulative frequencies.
   c. Arithmetic-decode one token from the payload.
   d. Append decoded token to prefix.
4. Decode token ids back to text.
5. Verify byte length and CRC32.
```

The core invariant is:

```text
At every token position, encoder and decoder must compute the same cumulative
frequency table.
```

If that invariant holds, arithmetic coding gives a lossless compressed payload.
If the model predicts the real text well, the payload becomes short.
