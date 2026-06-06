from __future__ import annotations

import hashlib
import math
import struct
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

from .arithmetic import ArithmeticDecoder, ArithmeticEncoder, BitReader, BitWriter, frequencies_to_cumulative
from .model import GPT, load_baseline_model


MAGIC = b"LLMZ3"
DEFAULT_TOTAL_FREQ = 1 << 20
DEFAULT_CONTEXT = 1024
_MODEL_CACHE: dict[tuple[str, str], GPT] = {}
_TOKENIZER_CACHE: dict[str, spm.SentencePieceProcessor] = {}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError(f"varint cannot encode negative value: {value}")
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def read_varint(f) -> int:
    shift = 0
    value = 0
    for _ in range(10):
        raw = f.read(1)
        if not raw:
            raise ValueError("Truncated varint")
        byte = raw[0]
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value
        shift += 7
    raise ValueError("varint is too long")


def logits_to_cumulative(logits: torch.Tensor, total_freq: int = DEFAULT_TOTAL_FREQ) -> list[int]:
    probs = torch.softmax(logits.float().cpu(), dim=-1).numpy().astype(np.float64)
    vocab_size = probs.shape[0]
    if total_freq <= vocab_size:
        raise ValueError(f"total_freq={total_freq} must exceed vocab_size={vocab_size}")

    scaled = probs * (total_freq - vocab_size)
    floors = np.floor(scaled).astype(np.int64)
    freqs = floors + 1
    remainder = int(total_freq - freqs.sum())
    if remainder > 0:
        frac = scaled - floors
        order = np.argsort(-frac, kind="stable")
        freqs[order[:remainder]] += 1
    elif remainder < 0:
        order = np.argsort(-freqs, kind="stable")
        need = -remainder
        for idx in order:
            if need == 0:
                break
            removable = min(int(freqs[idx] - 1), need)
            if removable > 0:
                freqs[idx] -= removable
                need -= removable
        if need:
            raise ValueError("Could not normalize frequency table without zeroing symbols")

    return frequencies_to_cumulative(freqs.tolist())


def payload_fraction(payload: bytes, prefix_bits: int = 64) -> dict:
    bit_count = len(payload) * 8
    if not payload:
        return {
            "bit_count": 0,
            "binary": "0.0",
            "decimal": "0",
            "marker": 0.0,
        }

    bits = "".join(f"{byte:08b}" for byte in payload)
    shown_bits = bits[: min(len(bits), 96)]
    decimal_bits = bits[: min(len(bits), prefix_bits)]
    numerator = int(decimal_bits, 2) if decimal_bits else 0
    denominator = 1 << len(decimal_bits)
    marker = numerator / denominator if denominator else 0.0
    decimal = f"{marker:.18f}".rstrip("0").rstrip(".")
    return {
        "bit_count": bit_count,
        "binary": f"0.{shown_bits}{'...' if len(bits) > len(shown_bits) else ''}",
        "decimal": decimal or "0",
        "marker": marker,
    }


def token_text(tokenizer: spm.SentencePieceProcessor, token: int) -> tuple[str, str]:
    try:
        piece = tokenizer.id_to_piece(token)
    except Exception:
        piece = str(token)
    try:
        text = tokenizer.decode([token])
    except Exception:
        text = piece
    return piece, text


class CachedLogitPredictor:
    def __init__(self, model: GPT, *, bos_id: int, max_context: int):
        self.model = model
        self.device = next(model.parameters()).device
        self.bos_id = bos_id
        self.max_context = max_context
        self.prefix = [bos_id]
        self.caches = None
        self.position = 0
        self._last_logits: torch.Tensor | None = None
        self._feed(bos_id)

    @torch.inference_mode()
    def _feed(self, token: int) -> torch.Tensor:
        input_id = torch.tensor([token], dtype=torch.long, device=self.device)
        logits, self.caches = self.model.forward_logits_step(input_id, self.caches, self.position)
        self.position += 1
        self._last_logits = logits[0]
        return self._last_logits

    def _rebuild(self) -> None:
        window = self.prefix[-self.max_context :]
        self.caches = None
        self.position = 0
        self._last_logits = None
        for token in window:
            self._feed(token)

    def next(self) -> torch.Tensor:
        if self._last_logits is None:
            raise RuntimeError("CachedLogitPredictor has not been initialized")
        return self._last_logits

    def append(self, token: int) -> None:
        self.prefix.append(token)
        if len(self.prefix) > self.max_context:
            self._rebuild()
        else:
            self._feed(token)


def load_tokenizer(tokenizer_path: str | Path) -> spm.SentencePieceProcessor:
    return spm.SentencePieceProcessor(model_file=str(tokenizer_path))


def get_cached_model(checkpoint_path: str | Path) -> GPT:
    path = str(Path(checkpoint_path).resolve())
    device = str(torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    key = (path, device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_baseline_model(path)
    return _MODEL_CACHE[key]


def get_cached_tokenizer(tokenizer_path: str | Path) -> spm.SentencePieceProcessor:
    path = str(Path(tokenizer_path).resolve())
    if path not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[path] = load_tokenizer(path)
    return _TOKENIZER_CACHE[path]


def compress_bytes(
    data: bytes,
    model: GPT,
    tokenizer: spm.SentencePieceProcessor,
    *,
    model_path: str | Path,
    tokenizer_path: str | Path,
    total_freq: int = DEFAULT_TOTAL_FREQ,
    max_context: int = DEFAULT_CONTEXT,
    trace_limit: int = 0,
) -> tuple[bytes, dict, bytes]:
    text = data.decode("utf-8")
    tokens = tokenizer.encode(text, out_type=int)
    roundtrip_text = tokenizer.decode(tokens)
    normalized_data = roundtrip_text.encode("utf-8")
    if normalized_data != data:
        raise ValueError(
            "Tokenizer is not byte-exact for this input. Normalize the input whitespace "
            "or switch to a byte-level tokenizer before compressing."
        )
    bos_id = int(tokenizer.bos_id())
    if bos_id < 0:
        raise ValueError("Tokenizer must define a BOS token")

    writer = BitWriter()
    encoder = ArithmeticEncoder(writer)
    predictor = CachedLogitPredictor(model, bos_id=bos_id, max_context=max_context)
    trace_tokens = []
    for index, token in enumerate(tokens):
        token = int(token)
        cumulative = logits_to_cumulative(predictor.next(), total_freq)
        if index < trace_limit:
            sym_low = int(cumulative[token])
            sym_high = int(cumulative[token + 1])
            freq = sym_high - sym_low
            probability = freq / total_freq
            piece, text_for_token = token_text(tokenizer, token)
            trace_tokens.append({
                "index": index,
                "token_id": token,
                "piece": piece,
                "text": text_for_token,
                "start": sym_low / total_freq,
                "end": sym_high / total_freq,
                "frequency": freq,
                "probability": probability,
                "bits": -math.log2(probability),
            })
        encoder.encode(token, cumulative)
        predictor.append(token)
    payload = encoder.finish()
    header = {
        "version": 3,
        "token_count": len(tokens),
        "crc32": zlib.crc32(data) & 0xFFFFFFFF,
        "total_freq": total_freq,
        "max_context": max_context,
        "bos_id": bos_id,
    }
    if trace_limit:
        header["interval_trace"] = {
            "token_count": len(tokens),
            "shown_count": len(trace_tokens),
            "total_freq": total_freq,
            "payload_fraction": payload_fraction(payload),
            "tokens": trace_tokens,
        }
    return payload, header, b""


def decompress_bytes(
    payload: bytes,
    header: dict,
    model: GPT,
    tokenizer: spm.SentencePieceProcessor,
) -> bytes:
    total_freq = int(header["total_freq"])
    max_context = int(header["max_context"])
    token_count = int(header["token_count"])
    bos_id = int(header["bos_id"])

    reader = BitReader(payload)
    decoder = ArithmeticDecoder(reader)
    predictor = CachedLogitPredictor(model, bos_id=bos_id, max_context=max_context)

    tokens: list[int] = []
    for _ in range(token_count):
        cumulative = logits_to_cumulative(predictor.next(), total_freq)
        token = decoder.decode(cumulative)
        tokens.append(token)
        predictor.append(token)

    data = tokenizer.decode(tokens).encode("utf-8")
    crc32 = zlib.crc32(data) & 0xFFFFFFFF
    if crc32 != int(header["crc32"]):
        raise ValueError(f"Decoded CRC32 mismatch: {crc32} != {header['crc32']}")
    return data


def write_archive(path: str | Path, payload: bytes, header: dict, sidecar: bytes = b"") -> None:
    if sidecar:
        raise ValueError("LLMZ3 does not support sidecar data")
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(encode_varint(int(header["token_count"])))
        f.write(encode_varint(int(header["total_freq"])))
        f.write(encode_varint(int(header["max_context"])))
        f.write(encode_varint(int(header["bos_id"])))
        f.write(struct.pack(">I", int(header["crc32"])))
        f.write(payload)


def read_archive(path: str | Path) -> tuple[bytes, dict]:
    with open(path, "rb") as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(f"Not an llmzip archive: {path}")

        token_count = read_varint(f)
        total_freq = read_varint(f)
        max_context = read_varint(f)
        bos_id = read_varint(f)
        crc_raw = f.read(4)
        if len(crc_raw) != 4:
            raise ValueError(f"Truncated llmzip archive: {path}")
        header = {
            "version": 3,
            "token_count": token_count,
            "total_freq": total_freq,
            "max_context": max_context,
            "bos_id": bos_id,
            "crc32": struct.unpack(">I", crc_raw)[0],
        }
        payload = f.read()
    return payload, header


def compress_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    total_freq: int = DEFAULT_TOTAL_FREQ,
    max_context: int = DEFAULT_CONTEXT,
    trace_limit: int = 0,
) -> dict:
    model = get_cached_model(checkpoint_path)
    tokenizer = get_cached_tokenizer(tokenizer_path)
    data = Path(input_path).read_bytes()
    payload, header, sidecar = compress_bytes(
        data,
        model,
        tokenizer,
        model_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        total_freq=total_freq,
        max_context=max_context,
        trace_limit=trace_limit,
    )
    write_archive(output_path, payload, header, sidecar)
    archive_size = Path(output_path).stat().st_size
    header["archive_bytes"] = archive_size
    header["bpb_file"] = archive_size * 8 / max(len(data), 1)
    header["bpb_payload"] = len(payload) * 8 / max(len(data), 1)
    return header


def decompress_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
) -> dict:
    model = get_cached_model(checkpoint_path)
    tokenizer = get_cached_tokenizer(tokenizer_path)
    payload, header = read_archive(input_path)
    if "model_sha256" in header and header.get("model_sha256") != sha256_file(checkpoint_path):
        raise ValueError("Checkpoint hash does not match archive header")
    if "tokenizer_sha256" in header and header.get("tokenizer_sha256") != sha256_file(tokenizer_path):
        raise ValueError("Tokenizer hash does not match archive header")
    data = decompress_bytes(payload, header, model, tokenizer)
    Path(output_path).write_bytes(data)
    return dict(header)
