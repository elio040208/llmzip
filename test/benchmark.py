from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llmzip import compress_file, decompress_file

DEFAULT_CHECKPOINT = ROOT / "models" / "baseline_sp1024.pt"
DEFAULT_TOKENIZER = ROOT / "tokenizers" / "fineweb_1024_bpe.model"
DEFAULT_URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
DEFAULT_TEXT = ROOT / "test" / "data" / "clean_gutenberg_2k.txt"
OUT_DIR = ROOT / "test" / "out"


def normalize_text(data: bytes, limit_bytes: int) -> bytes:
    text = data.decode("utf-8", errors="ignore")
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start = text.find(start_marker)
    if start >= 0:
        text = text[text.find("\n", start) + 1 :]
    end = text.find(end_marker)
    if end >= 0:
        text = text[:end]

    text = " ".join(text.split())
    output = text.encode("utf-8")
    if len(output) <= limit_bytes:
        return output

    chunk = output[:limit_bytes]
    while chunk and (chunk[-1] & 0b1100_0000) == 0b1000_0000:
        chunk = chunk[:-1]
    last_space = chunk.rfind(b" ")
    if last_space > limit_bytes * 3 // 4:
        chunk = chunk[:last_space]
    return chunk


def prepare_clean_text(path: Path, url: str, limit_bytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read()
    path.write_bytes(normalize_text(data, limit_bytes))


def run_command(name: str, command: list[str], output_path: Path, original_size: int) -> dict:
    t0 = time.perf_counter()
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elapsed = time.perf_counter() - t0
    size = output_path.stat().st_size
    return {
        "name": name,
        "bytes": size,
        "bpb": size * 8 / original_size,
        "seconds": elapsed,
    }


def benchmark(args: argparse.Namespace) -> list[dict]:
    text_path = Path(args.input) if args.input else Path(args.output_text)
    if args.input is None:
        prepare_clean_text(text_path, args.url, args.limit_bytes)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    original = text_path.read_bytes()
    original_size = len(original)
    rows = [{
        "name": "raw",
        "bytes": original_size,
        "bpb": 8.0,
        "seconds": 0.0,
    }]

    llmz_path = OUT_DIR / f"{text_path.name}.llmz"
    restored_path = OUT_DIR / f"{text_path.name}.restored"
    print(f"Running llmzip on {original_size} bytes from {text_path}...", flush=True)
    t0 = time.perf_counter()
    llm_meta = compress_file(
        text_path,
        llmz_path,
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        total_freq=args.total_freq,
        max_context=args.max_context,
    )
    encode_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    decompress_file(
        llmz_path,
        restored_path,
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
    )
    decode_seconds = time.perf_counter() - t1
    if restored_path.read_bytes() != original:
        raise RuntimeError("llmzip restored bytes do not match original")
    rows.append({
        "name": "llmzip_file",
        "bytes": llmz_path.stat().st_size,
        "bpb": llmz_path.stat().st_size * 8 / original_size,
        "seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "payload_bpb": llm_meta["bpb_payload"],
    })

    commands = []
    gzip_path = OUT_DIR / f"{text_path.name}.gz"
    commands.append(("gzip-9", ["gzip", "-c", "-9", str(text_path)], gzip_path))
    bzip2_path = OUT_DIR / f"{text_path.name}.bz2"
    commands.append(("bzip2-9", ["bzip2", "-c", "-9", str(text_path)], bzip2_path))
    if shutil.which("zstd"):
        zstd_path = OUT_DIR / f"{text_path.name}.zst"
        commands.append(("zstd-19", ["zstd", "-q", "-c", "-19", str(text_path)], zstd_path))
    if shutil.which("brotli"):
        br_path = OUT_DIR / f"{text_path.name}.br"
        commands.append(("brotli-11", ["brotli", "-c", "-q", "11", str(text_path)], br_path))
    if shutil.which("xz"):
        xz_path = OUT_DIR / f"{text_path.name}.xz"
        commands.append(("xz-9", ["xz", "-c", "-9", str(text_path)], xz_path))

    for name, command, output_path in commands:
        print(f"Running {name}...", flush=True)
        with open(output_path, "wb") as f:
            t0 = time.perf_counter()
            subprocess.run(command, check=True, stdout=f, stderr=subprocess.DEVNULL)
            elapsed = time.perf_counter() - t0
        size = output_path.stat().st_size
        rows.append({
            "name": name,
            "bytes": size,
            "bpb": size * 8 / original_size,
            "seconds": elapsed,
        })

    rows.sort(key=lambda row: row["bytes"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Optional local UTF-8 text file. Generates a clean benchmark text if omitted.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--limit-bytes", type=int, default=2 * 1024)
    parser.add_argument("--output-text", default=str(DEFAULT_TEXT), help="Path for generated clean benchmark text.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--total-freq", type=int, default=1 << 20)
    parser.add_argument("--max-context", type=int, default=1024)
    parser.add_argument("--csv", default=str(ROOT / "test" / "benchmark_results.csv"))
    args = parser.parse_args()

    rows = benchmark(args)
    print(json.dumps(rows, indent=2))
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
