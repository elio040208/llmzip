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


MAGIC_V1 = b"LLMZ1"
MAGIC = b"LLMZ2"
DEFAULT_TOTAL_FREQ = 1 << 20
DEFAULT_CONTEXT = 1024
FLAG_CRC32 = 1 << 0
LOGIT_MODE_TO_ID = {
    "full-prefix-v1": 0,
    "kv-cache-v1": 1,
}
ID_TO_LOGIT_MODE = {value: key for key, value in LOGIT_MODE_TO_ID.items()}
_MODEL_CACHE: dict[tuple[str, str], GPT] = {}
_TOKENIZER_CACHE: dict[str, spm.SentencePieceProcessor] = {}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


@torch.inference_mode()
def next_logits(model: GPT, prefix: list[int], device: torch.device, max_context: int) -> torch.Tensor:
    if not prefix:
        raise ValueError("prefix must contain at least one token")
    window = prefix[-max_context:]
    input_ids = torch.tensor([window], dtype=torch.long, device=device)
    return model.forward_logits(input_ids)[0, -1]


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


class LegacyLogitPredictor:
    def __init__(self, model: GPT, *, bos_id: int, max_context: int):
        self.model = model
        self.device = next(model.parameters()).device
        self.prefix = [bos_id]
        self.max_context = max_context

    def next(self) -> torch.Tensor:
        return next_logits(self.model, self.prefix, self.device, self.max_context)

    def append(self, token: int) -> None:
        self.prefix.append(token)


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
    if normalized_data != data:
        raise ValueError(
            "Tokenizer is not byte-exact for this input. Normalize the input whitespace "
            "or switch to a byte-level tokenizer before compressing."
        )
    sidecar = b""
    bos_id = int(tokenizer.bos_id())
    if bos_id < 0:
        raise ValueError("Tokenizer must define a BOS token")

    writer = BitWriter()
    encoder = ArithmeticEncoder(writer)
    logit_mode = "kv-cache-v1"
    predictor = CachedLogitPredictor(model, bos_id=bos_id, max_context=max_context)
    for token in tokens:
        cumulative = logits_to_cumulative(predictor.next(), total_freq)
        encoder.encode(int(token), cumulative)
        predictor.append(int(token))
    payload = encoder.finish()
    header = {
        "version": 2,
        "encoding": "utf-8",
        "token_count": len(tokens),
        "byte_count": len(data),
        "crc32": zlib.crc32(data) & 0xFFFFFFFF,
        "total_freq": total_freq,
        "max_context": max_context,
        "bos_id": bos_id,
        "logit_mode": logit_mode,
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

    def decode_with_mode(logit_mode: str) -> bytes:
        reader = BitReader(payload)
        decoder = ArithmeticDecoder(reader)
        if logit_mode == "kv-cache-v1":
            predictor = CachedLogitPredictor(model, bos_id=bos_id, max_context=max_context)
        elif logit_mode == "full-prefix-v1":
            predictor = LegacyLogitPredictor(model, bos_id=bos_id, max_context=max_context)
        else:
            raise ValueError(f"Unsupported logit_mode: {logit_mode}")

        tokens: list[int] = []
        for _ in range(token_count):
            cumulative = logits_to_cumulative(predictor.next(), total_freq)
            token = decoder.decode(cumulative)
            tokens.append(token)
            predictor.append(token)

        text = tokenizer.decode(tokens)
        normalized_data = text.encode("utf-8")
        if "normalized_byte_count" in header and len(normalized_data) != int(header["normalized_byte_count"]):
            raise ValueError(
                f"Decoded normalized byte length mismatch: {len(normalized_data)} != {header['normalized_byte_count']}"
            )
        normalized_digest = sha256_bytes(normalized_data) if "normalized_sha256" in header else None
        if "normalized_sha256" in header and normalized_digest != header["normalized_sha256"]:
            raise ValueError(f"Decoded normalized SHA256 mismatch: {normalized_digest} != {header['normalized_sha256']}")
        data = apply_sidecar(normalized_data, header.get("_sidecar", b""))
        if len(data) != int(header["byte_count"]):
            raise ValueError(f"Decoded byte length mismatch: {len(data)} != {header['byte_count']}")
        if "crc32" in header:
            crc32 = zlib.crc32(data) & 0xFFFFFFFF
            if crc32 != int(header["crc32"]):
                raise ValueError(f"Decoded CRC32 mismatch: {crc32} != {header['crc32']}")
        digest = sha256_bytes(data) if "sha256" in header else None
        if "sha256" in header and digest != header["sha256"]:
            raise ValueError(f"Decoded SHA256 mismatch: {digest} != {header['sha256']}")
        return data

    if "logit_mode" in header:
        return decode_with_mode(str(header["logit_mode"]))

    errors = []
    for logit_mode in ("full-prefix-v1", "kv-cache-v1"):
        try:
            return decode_with_mode(logit_mode)
        except ValueError as exc:
            errors.append(f"{logit_mode}: {exc}")
    raise ValueError("Could not decode archive with any legacy logit mode: " + "; ".join(errors))


def write_archive(path: str | Path, payload: bytes, header: dict, sidecar: bytes = b"") -> None:
    if sidecar:
        raise ValueError("LLMZ2 does not support sidecar data")
    header = dict(header)
    header["payload_bytes"] = len(payload)
    logit_mode = str(header["logit_mode"])
    if logit_mode not in LOGIT_MODE_TO_ID:
        raise ValueError(f"Unsupported logit_mode: {logit_mode}")
    flags = FLAG_CRC32 if "crc32" in header else 0
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(bytes([flags, LOGIT_MODE_TO_ID[logit_mode]]))
        f.write(encode_varint(int(header["token_count"])))
        f.write(encode_varint(int(header["byte_count"])))
        f.write(encode_varint(len(payload)))
        f.write(encode_varint(int(header["total_freq"])))
        f.write(encode_varint(int(header["max_context"])))
        f.write(encode_varint(int(header["bos_id"])))
        if flags & FLAG_CRC32:
            f.write(struct.pack(">I", int(header["crc32"])))
        f.write(payload)


def read_archive(path: str | Path) -> tuple[bytes, dict]:
    with open(path, "rb") as f:
        magic = f.read(len(MAGIC))
        if magic == MAGIC_V1:
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

        if magic != MAGIC:
            raise ValueError(f"Not an llmzip archive: {path}")

        flags_raw = f.read(1)
        logit_mode_raw = f.read(1)
        if not flags_raw or not logit_mode_raw:
            raise ValueError(f"Truncated llmzip archive: {path}")
        flags = flags_raw[0]
        logit_mode_id = logit_mode_raw[0]
        if logit_mode_id not in ID_TO_LOGIT_MODE:
            raise ValueError(f"Unsupported logit mode id: {logit_mode_id}")

        token_count = read_varint(f)
        byte_count = read_varint(f)
        payload_len = read_varint(f)
        total_freq = read_varint(f)
        max_context = read_varint(f)
        bos_id = read_varint(f)
        header = {
            "version": 2,
            "encoding": "utf-8",
            "token_count": token_count,
            "byte_count": byte_count,
            "payload_bytes": payload_len,
            "sidecar_bytes": 0,
            "total_freq": total_freq,
            "max_context": max_context,
            "bos_id": bos_id,
            "logit_mode": ID_TO_LOGIT_MODE[logit_mode_id],
            "_sidecar": b"",
        }
        if flags & FLAG_CRC32:
            crc_raw = f.read(4)
            if len(crc_raw) != 4:
                raise ValueError(f"Truncated llmzip archive: {path}")
            header["crc32"] = struct.unpack(">I", crc_raw)[0]
        unknown_flags = flags & ~FLAG_CRC32
        if unknown_flags:
            raise ValueError(f"Unsupported archive flags: {unknown_flags}")
        payload = f.read(payload_len)
        if len(payload) != payload_len:
            raise ValueError(f"Truncated llmzip archive: {path}")
        if f.read(1):
            raise ValueError(f"Unexpected trailing bytes in llmzip archive: {path}")
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
    )
    write_archive(output_path, payload, header, sidecar)
    archive_size = Path(output_path).stat().st_size
    header["archive_bytes"] = archive_size
    header["payload_bytes"] = len(payload)
    header["sidecar_bytes"] = 0
    header["bpb_file"] = archive_size * 8 / max(len(data), 1)
    header["bpb_payload"] = len(payload) * 8 / max(len(data), 1)
    header["bpb_payload_plus_sidecar"] = len(payload) * 8 / max(len(data), 1)
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
    public_header = dict(header)
    public_header.pop("_sidecar", None)
    return public_header
