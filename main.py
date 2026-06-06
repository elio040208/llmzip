from __future__ import annotations

import argparse
import base64
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from llmzip import compress_file, decompress_file
from llmzip.codec import read_archive


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
DEFAULT_CHECKPOINT = ROOT / "models" / "baseline_sp1024.pt"
DEFAULT_TOKENIZER = ROOT / "tokenizers" / "fineweb_1024_bpe.model"

checkpoint_path = DEFAULT_CHECKPOINT
tokenizer_path = DEFAULT_TOKENIZER

app = FastAPI(title="llmzip web")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CompressRequest(BaseModel):
    text: str = ""
    total_freq: int = Field(default=1 << 20, ge=2048)
    max_context: int = Field(default=1024, ge=8)


class DecompressRequest(BaseModel):
    archive_base64: str


def hex_preview(data: bytes, limit: int = 4096) -> str:
    shown = data[:limit]
    lines = []
    for offset in range(0, len(shown), 16):
        chunk = shown[offset : offset + 16]
        hex_part = " ".join(f"{byte:02x}" for byte in chunk)
        ascii_part = "".join(chr(byte) if 32 <= byte < 127 else "." for byte in chunk)
        lines.append(f"{offset:08x}  {hex_part:<47}  {ascii_part}")
    if len(data) > limit:
        lines.append(f"... truncated, showing {limit} of {len(data)} bytes")
    return "\n".join(lines)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.post("/api/compress")
def compress_endpoint(request: CompressRequest) -> dict:
    data = request.text.encode("utf-8")
    try:
        with tempfile.TemporaryDirectory(prefix="llmzip-web-") as tmp:
            input_path = Path(tmp) / "input.txt"
            archive_path = Path(tmp) / "output.llmz"
            input_path.write_bytes(data)
            meta = compress_file(
                input_path,
                archive_path,
                checkpoint_path=checkpoint_path,
                tokenizer_path=tokenizer_path,
                total_freq=request.total_freq,
                max_context=request.max_context,
            )
            archive = archive_path.read_bytes()
            _, parsed_header = read_archive(archive_path)
            parsed_header.pop("_sidecar", None)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "archive_base64": base64.b64encode(archive).decode("ascii"),
        "archive_hex": hex_preview(archive),
        "archive_bytes": len(archive),
        "input_bytes": len(data),
        "metadata": meta,
        "parsed_header": parsed_header,
    }


@app.post("/api/decompress")
def decompress_endpoint(request: DecompressRequest) -> dict:
    try:
        archive = base64.b64decode(request.archive_base64.strip(), validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 archive") from exc

    try:
        with tempfile.TemporaryDirectory(prefix="llmzip-web-") as tmp:
            archive_path = Path(tmp) / "input.llmz"
            output_path = Path(tmp) / "output.txt"
            archive_path.write_bytes(archive)
            meta = decompress_file(
                archive_path,
                output_path,
                checkpoint_path=checkpoint_path,
                tokenizer_path=tokenizer_path,
            )
            data = output_path.read_bytes()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "text": data.decode("utf-8"),
        "output_bytes": len(data),
        "metadata": meta,
    }


app.mount("/", StaticFiles(directory=WEB_ROOT, html=True), name="web")


def main() -> None:
    global checkpoint_path, tokenizer_path
    parser = argparse.ArgumentParser(prog="llmzip-web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)
    print(f"llmzip web UI: http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
