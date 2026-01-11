#!/usr/bin/env python3
import io
import os
from pathlib import Path

from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from inference.train_hybrid_16millions import ocr_pil
from inference.new_training import ocr_pil2

app = FastAPI(title="Text Recognition API (no torchvision)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

INDEX_PATH = Path(__file__).with_name("index.html")


@app.get("/", response_class=HTMLResponse)
async def index():
    if INDEX_PATH.exists():
        return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<html><body><h3>index.html not found</h3><p>Put index.html next to main.py</p></body></html>",
        status_code=200,
    )


def _read_image_or_400(data: bytes) -> Image.Image:
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")
    data = await file.read()
    img = _read_image_or_400(data)
    text = ocr_pil(img)
    return {"text": text}


@app.post("/recognize2")
async def recognize2(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")
    data = await file.read()
    img = _read_image_or_400(data)
    text = ocr_pil2(img)
    return {"text": text}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5348, log_level="info")
