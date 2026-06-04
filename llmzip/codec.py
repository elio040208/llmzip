from __future__ import annotations

import hashlib
import json
import struct
import zlib
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

from .arithmetic import ArithmeticDecoder, ArithmeticEncoder, BitReader, BitWriter, frequencies_to_cumulative
from .model import GPT, load_baseline_model


MAGIC = b"LLMZ1"
DEFAULT_TOTAL_FREQ = 1 << 20
DEFAULT_CONTEXT = 1024


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


@torch.inference_mode()
def next_logits(model: GPT, prefix: list[int], device: torch.device, max_context: int) -> torch.Tensor:
    if not prefix:
        raise ValueError("prefix must contain at least one token")
    window = prefix[-max_context:]
    input_ids = torch.tensor([window], dtype=torch.long, device=device)
    return model.forward_logits(input_ids)[0, -1]


def load_tokenizer(tokenizer_path: str | Path) -> spm.SentencePieceProcessor:
    return spm.SentencePieceProcessor(model_file=str(tokenizer_path))


def make_sidecar(source: bytes, target: bytes) -> bytes:
    """Return compressed edit script that transforms source bytes into target bytes."""
    if source == target:
        return b""
    matcher = SequenceMatcher(None, source, target, autojunk=False)
    edits = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        edits.append([tag, i1, i2, list(target[j1:j2])])
    raw = json.dumps(edits, separators=(",", ":")).encode("utf-8")
    return zlib.compress(raw, level=9)


def apply_sidecar(source: bytes, sidecar: bytes) -> bytes:
    if not sidecar:
        return source
    edits = json.loads(zlib.decompress(sidecar).decode("utf-8"))
    out = bytearray()
    cursor = 0
    for tag, i1, i2, replacement in edits:
        out.extend(source[cursor:i1])
        if tag in {"insert", "replace"}:
            out.extend(bytes(replacement))
        elif tag != "delete":
            raise ValueError(f"Unknown sidecar edit tag: {tag}")
        cursor = i2
    out.extend(source[cursor:])
    return bytes(out)


def compress_bytes(
    data: bytes,
    model: GPT,
    tokenizer: spm.SentencePieceProcessor,
    *,
    model_path: str | Path,
    tokenizer_path: str | Path,
    total_freq: int = DEFAULT_TOTAL_FREQ,
    max_context: int = DEFAULT_CONTEXT,
) -> tuple[bytes, dict]:
    text = data.decode("utf-8")
    tokens = tokenizer.encode(text, out_type=int)
    roundtrip_text = tokenizer.decode(tokens)
    normalized_data = roundtrip_text.encode("utf-8")
    sidecar = make_sidecar(normalized_data, data)
    bos_id = int(tokenizer.bos_id())
    if bos_id < 0:
        raise ValueError("Tokenizer must define a BOS token")

    writer = BitWriter()
    encoder = ArithmeticEncoder(writer)
    device = next(model.parameters()).device
    prefix = [bos_id]
    for token in tokens:
        cumulative = logits_to_cumulative(next_logits(model, prefix, device, max_context), total_freq)
        encoder.encode(int(token), cumulative)
        prefix.append(int(token))
    payload = encoder.finish()
    header = {
        "version": 1,
        "encoding": "utf-8",
        "token_count": len(tokens),
        "byte_count": len(data),
        "normalized_byte_count": len(normalized_data),
        "sha256": sha256_bytes(data),
        "normalized_sha256": sha256_bytes(normalized_data),
        "model_sha256": sha256_file(model_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "total_freq": total_freq,
        "max_context": max_context,
        "bos_id": bos_id,
        "sidecar_codec": "zlib-json-diff-v1" if sidecar else "none",
    }
    return payload, header, sidecar


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
    device = next(model.parameters()).device
    prefix = [bos_id]
    tokens: list[int] = []
    for _ in range(token_count):
        cumulative = logits_to_cumulative(next_logits(model, prefix, device, max_context), total_freq)
        token = decoder.decode(cumulative)
        tokens.append(token)
        prefix.append(token)

    text = tokenizer.decode(tokens)
    normalized_data = text.encode("utf-8")
    if len(normalized_data) != int(header["normalized_byte_count"]):
        raise ValueError(
            f"Decoded normalized byte length mismatch: {len(normalized_data)} != {header['normalized_byte_count']}"
        )
    normalized_digest = sha256_bytes(normalized_data)
    if normalized_digest != header["normalized_sha256"]:
        raise ValueError(f"Decoded normalized SHA256 mismatch: {normalized_digest} != {header['normalized_sha256']}")
    data = apply_sidecar(normalized_data, header.get("_sidecar", b""))
    if len(data) != int(header["byte_count"]):
        raise ValueError(f"Decoded byte length mismatch: {len(data)} != {header['byte_count']}")
    digest = sha256_bytes(data)
    if digest != header["sha256"]:
        raise ValueError(f"Decoded SHA256 mismatch: {digest} != {header['sha256']}")
    return data


def write_archive(path: str | Path, payload: bytes, header: dict, sidecar: bytes = b"") -> None:
    header = dict(header)
    header["payload_bytes"] = len(payload)
    header["sidecar_bytes"] = len(sidecar)
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack(">I", len(header_bytes)))
        f.write(header_bytes)
        f.write(payload)
        f.write(sidecar)


def read_archive(path: str | Path) -> tuple[bytes, dict]:
    with open(path, "rb") as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(f"Not an llmzip archive: {path}")
        header_len = struct.unpack(">I", f.read(4))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        payload_len = int(header.get("payload_bytes", -1))
        sidecar_len = int(header.get("sidecar_bytes", 0))
        if payload_len < 0:
            payload = f.read()
            sidecar = b""
        else:
            payload = f.read(payload_len)
            sidecar = f.read(sidecar_len)
            if len(payload) != payload_len or len(sidecar) != sidecar_len:
                raise ValueError(f"Truncated llmzip archive: {path}")
        header["_sidecar"] = sidecar
    return payload, header


def compress_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    total_freq: int = DEFAULT_TOTAL_FREQ,
    max_context: int = DEFAULT_CONTEXT,
) -> dict:
    model = load_baseline_model(checkpoint_path)
    tokenizer = load_tokenizer(tokenizer_path)
    data = Path(input_path).read_bytes()
    payload, header, sidecar = compress_bytes(
        data,
        model,
        tokenizer,
        model_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        total_freq=total_freq,
        max_context=max_context,
    )
    write_archive(output_path, payload, header, sidecar)
    archive_size = Path(output_path).stat().st_size
    header["archive_bytes"] = archive_size
    header["payload_bytes"] = len(payload)
    header["sidecar_bytes"] = len(sidecar)
    header["bpb_file"] = archive_size * 8 / max(len(data), 1)
    header["bpb_payload"] = len(payload) * 8 / max(len(data), 1)
    header["bpb_payload_plus_sidecar"] = (len(payload) + len(sidecar)) * 8 / max(len(data), 1)
    return header


def decompress_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
) -> dict:
    model = load_baseline_model(checkpoint_path)
    tokenizer = load_tokenizer(tokenizer_path)
    payload, header = read_archive(input_path)
    if header.get("model_sha256") != sha256_file(checkpoint_path):
        raise ValueError("Checkpoint hash does not match archive header")
    if header.get("tokenizer_sha256") != sha256_file(tokenizer_path):
        raise ValueError("Tokenizer hash does not match archive header")
    data = decompress_bytes(payload, header, model, tokenizer)
    Path(output_path).write_bytes(data)
    public_header = dict(header)
    public_header.pop("_sidecar", None)
    return public_header
