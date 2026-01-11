#!/usr/bin/env python3
import io
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference.train_hybrid_16millions import ocr_pil

# -----------------------
# FastAPI
# -----------------------
app = FastAPI(title="Text Recognition API (no torchvision)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    text = ocr_pil(img)
    return {"text": text}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("test:app", host="0.0.0.0", port=5348, log_level="info")
