#!/usr/bin/env python3
import io
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps


# -----------------------
# Config (hard-coded) - UPDATED to match trainer v2
# -----------------------
@dataclass
class CFG:
    CKPT_PATH: str = "./model/best3.pt"
    VOCAB_JSON: str = "./model/vocab_char.json"

    IMG_H: int = 48
    IMG_W: int = 640

    ENC_DIM: int = 256
    ENC_LAYERS: int = 4
    ENC_HEADS: int = 8
    ENC_FF: int = 1024
    DROPOUT: float = 0.15

    USE_DECODER: bool = True
    DEC_DIM: int = 256
    DEC_LAYERS: int = 3
    DEC_HEADS: int = 8
    DEC_FF: int = 1024
    MAX_DEC_LEN: int = 260

    USE_CTC: bool = True

    USE_LM: bool = True
    LM_DIM: int = 224
    LM_LAYERS: int = 2
    LM_HEADS: int = 7
    LM_FF: int = 896

    BEAM: int = 6
    BEAM_LENP: float = 0.75
    USE_LM_FUSION_EVAL: bool = True
    LM_FUSION_ALPHA: float = 0.25

    EOS_LOGP_BIAS: float = 0.55
    EOS_BIAS_UNTIL_LEN: int = 28

    UNK_TOKEN: str = "<unk>"
    COLLAPSE_WHITESPACE: bool = True
    UNICODE_NFC: bool = True


# -----------------------
# Tokenizer (UPDATED: NFC normalization)
# -----------------------
class CharTokenizer:
    def __init__(self, vocab_json: str, unk_token: str = "<unk>", collapse_whitespace: bool = True, unicode_nfc: bool = True):
        with open(vocab_json, "r", encoding="utf-8") as f:
            vocab_raw: Dict[str, int] = json.load(f)

        if unk_token not in vocab_raw:
            vocab_raw[unk_token] = max(vocab_raw.values(), default=-1) + 1

        items = sorted(vocab_raw.items(), key=lambda kv: kv[1])
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        for new_id, (tok, _old) in enumerate(items):
            self.token_to_id[tok] = new_id
            self.id_to_token[new_id] = tok

        self.unk_token = unk_token
        self.unk_id = self.token_to_id[unk_token]
        self.collapse_whitespace = collapse_whitespace
        self.unicode_nfc = unicode_nfc

        self.blank_id = 0
        self.pad_id = 1
        self.ctc_offset = 2
        self.vocab_size = len(self.token_to_id)
        self.ctc_classes = self.vocab_size + self.ctc_offset

        self.dec_pad = 0
        self.dec_bos = 1
        self.dec_eos = 2
        self.dec_offset = 3
        self.dec_vocab = self.vocab_size + self.dec_offset

    def _norm(self, s: str) -> str:
        if s is None:
            return ""
        s = s.strip()
        if self.unicode_nfc:
            s = unicodedata.normalize("NFC", s)
        if self.collapse_whitespace:
            s = re.sub(r"\s+", " ", s).strip()
        return s

    def decode_dec(self, ids: List[int]) -> str:
        out = []
        for x in ids:
            if x in (self.dec_pad, self.dec_bos, self.dec_eos):
                continue
            y = x - self.dec_offset
            if 0 <= y < self.vocab_size:
                t = self.id_to_token.get(y, self.unk_token)
                out.append("�" if t == self.unk_token else t)
        return self._norm("".join(out))


# -----------------------
# Resize (NO crop)
# -----------------------
class ResizeKeepRatioPadNoCrop:
    def __init__(self, h: int, w: int):
        self.h = h
        self.w = w

    def __call__(self, img: Image.Image) -> Image.Image:
        iw, ih = img.size
        if ih <= 0 or iw <= 0:
            return img.resize((self.w, self.h), Image.BILINEAR)

        scale = self.h / float(ih)
        nw = max(1, int(round(iw * scale)))
        img = img.resize((nw, self.h), Image.BILINEAR)

        if nw == self.w:
            return img
        if nw < self.w:
            pad_total = self.w - nw
            left = pad_total // 2
            right = pad_total - left
            return ImageOps.expand(img, border=(left, 0, right, 0), fill=255)

        return img.resize((self.w, self.h), Image.BILINEAR)


# -----------------------
# Model (UPDATED to match trainer v2)
# -----------------------
class PosEnc1D(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0).to(dtype=x.dtype, device=x.device)


import math


class ConvStem(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        d = dropout
        self.net = nn.Sequential(
            nn.Conv2d(1, 48, 3, 1, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),

            nn.Conv2d(48, 96, 3, (2, 2), 1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),

            nn.Conv2d(96, 160, 3, (2, 2), 1, bias=False),
            nn.BatchNorm2d(160),
            nn.SiLU(inplace=True),

            nn.Conv2d(160, dim, 3, (2, 1), 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),

            nn.Dropout2d(d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridContextOCRV2(nn.Module):
    def __init__(self, cfg: CFG, tok: CharTokenizer):
        super().__init__()
        self.cfg = cfg
        self.tok = tok
        d = cfg.DROPOUT

        self.stem = ConvStem(cfg.ENC_DIM, d)
        self.enc_ln_in = nn.LayerNorm(cfg.ENC_DIM)
        self.pos = PosEnc1D(cfg.ENC_DIM)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.ENC_DIM,
            nhead=cfg.ENC_HEADS,
            dim_feedforward=cfg.ENC_FF,
            dropout=d,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=cfg.ENC_LAYERS)
        self.enc_ln = nn.LayerNorm(cfg.ENC_DIM)

        self.use_ctc = cfg.USE_CTC
        if self.use_ctc:
            self.ctc_head = nn.Sequential(
                nn.LayerNorm(cfg.ENC_DIM),
                nn.Dropout(d),
                nn.Linear(cfg.ENC_DIM, tok.ctc_classes),
            )

        self.use_decoder = cfg.USE_DECODER
        if self.use_decoder:
            self.mem_proj = nn.Linear(cfg.ENC_DIM, cfg.DEC_DIM, bias=False)
            self.dec_emb = nn.Embedding(tok.dec_vocab, cfg.DEC_DIM)
            dec_layer = nn.TransformerDecoderLayer(
                d_model=cfg.DEC_DIM,
                nhead=cfg.DEC_HEADS,
                dim_feedforward=cfg.DEC_FF,
                dropout=d,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.dec = nn.TransformerDecoder(dec_layer, num_layers=cfg.DEC_LAYERS)
            self.dec_ln = nn.LayerNorm(cfg.DEC_DIM)
            self.dec_head = nn.Linear(cfg.DEC_DIM, tok.dec_vocab)

        self.use_lm = cfg.USE_LM
        if self.use_lm:
            self.lm_emb = nn.Embedding(tok.dec_vocab, cfg.LM_DIM)
            lm_layer = nn.TransformerEncoderLayer(
                d_model=cfg.LM_DIM,
                nhead=cfg.LM_HEADS,
                dim_feedforward=cfg.LM_FF,
                dropout=d,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.lm = nn.TransformerEncoder(lm_layer, num_layers=cfg.LM_LAYERS)
            self.lm_ln = nn.LayerNorm(cfg.LM_DIM)
            self.lm_head = nn.Linear(cfg.LM_DIM, tok.dec_vocab)

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        x = self.stem(imgs)  # [B,C,H',W']
        x = F.adaptive_avg_pool2d(x, (1, x.size(-1)))  # [B,C,1,W']
        x = x.squeeze(2).permute(0, 2, 1)  # [B,T,C]
        x = self.enc_ln_in(x)
        x = self.pos(x)
        x = self.enc(x)
        x = self.enc_ln(x)
        return x

    @torch.no_grad()
    def lm_next_logp(self, prefix: torch.Tensor) -> torch.Tensor:
        L = prefix.size(1)
        causal = torch.triu(torch.ones((L, L), device=prefix.device, dtype=torch.bool), diagonal=1)
        x = self.lm_emb(prefix)
        x = self.lm(x, mask=causal)
        x = self.lm_ln(x)
        logits = self.lm_head(x)[:, -1, :]
        return F.log_softmax(logits, dim=-1).squeeze(0)


@torch.no_grad()
def beam_decode_one(model: HybridContextOCRV2, mem_proj_1: torch.Tensor, tok: CharTokenizer, cfg: CFG) -> str:
    device = mem_proj_1.device
    beams = [(0.0, [tok.dec_bos], False)]

    for _ in range(cfg.MAX_DEC_LEN):
        new_beams = []
        for score, seq, fin in beams:
            if fin:
                new_beams.append((score, seq, True))
                continue

            inp = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
            tgt = model.dec_emb(inp)
            L = inp.size(1)
            causal = torch.triu(torch.ones((L, L), device=device, dtype=torch.bool), diagonal=1)

            out = model.dec(tgt=tgt, memory=mem_proj_1, tgt_mask=causal)
            out = model.dec_ln(out)
            logits = model.dec_head(out)[:, -1, :]
            logp = F.log_softmax(logits, dim=-1).squeeze(0)

            if cfg.USE_LM and cfg.USE_LM_FUSION_EVAL:
                logp = logp + cfg.LM_FUSION_ALPHA * model.lm_next_logp(inp)

            cur_len = len(seq) - 1
            if cur_len < cfg.EOS_BIAS_UNTIL_LEN:
                logp[tok.dec_eos] = logp[tok.dec_eos] - (cfg.EOS_LOGP_BIAS * 1.6)
            else:
                logp[tok.dec_eos] = logp[tok.dec_eos] - cfg.EOS_LOGP_BIAS

            topv, topi = torch.topk(logp, k=cfg.BEAM)
            for v, i in zip(topv.tolist(), topi.tolist()):
                ns = seq + [int(i)]
                nf = (int(i) == tok.dec_eos)
                new_beams.append((score + float(v), ns, nf))

        def normed(s, seq_):
            L2 = max(1, len(seq_) - 1)
            return s / (L2 ** cfg.BEAM_LENP)

        new_beams.sort(key=lambda x: normed(x[0], x[1]), reverse=True)
        beams = new_beams[: cfg.BEAM]

        if all(b[2] for b in beams):
            break

    best = max(beams, key=lambda x: x[0] / (max(1, len(x[1]) - 1) ** cfg.BEAM_LENP))[1]
    ids = []
    for x in best[1:]:
        if x == tok.dec_eos:
            break
        ids.append(x)
    return tok.decode_dec(ids)


# -----------------------
# Preprocess without torchvision
# -----------------------
def pil_to_norm_tensor_gray(img: Image.Image) -> torch.Tensor:
    g = img.convert("L")
    arr = np.array(g, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    t = torch.from_numpy(arr).unsqueeze(0)
    return t


# -----------------------
# Global init (loads once)
# -----------------------
_cfg = CFG()
_device = "cuda" if torch.cuda.is_available() else "cpu"

_tok = CharTokenizer(
    _cfg.VOCAB_JSON,
    unk_token=_cfg.UNK_TOKEN,
    collapse_whitespace=_cfg.COLLAPSE_WHITESPACE,
    unicode_nfc=_cfg.UNICODE_NFC,
)
_model = HybridContextOCRV2(_cfg, _tok).to(_device)

_ckpt = torch.load(_cfg.CKPT_PATH, map_location="cpu")
_state = _ckpt.get("model", _ckpt)
_model.load_state_dict(_state, strict=True)

_ema_shadow = _ckpt.get("ema", None)
if isinstance(_ema_shadow, dict):
    msd = _model.state_dict()
    out = {}
    for k, v in msd.items():
        s = _ema_shadow[k]
        out[k] = s.to(device=v.device, dtype=v.dtype) if torch.is_floating_point(v) else s.to(device=v.device)
    _model.load_state_dict(out, strict=True)

_model.eval()

_pre = ResizeKeepRatioPadNoCrop(_cfg.IMG_H, _cfg.IMG_W)


@torch.no_grad()
def ocr_pil2(img: Image.Image) -> str:
    img = img.convert("RGB")
    img = _pre(img)
    x = pil_to_norm_tensor_gray(img).unsqueeze(0).to(_device)  # 1,1,H,W
    mem = _model.encode(x)
    mem_proj = _model.mem_proj(mem)
    return beam_decode_one(_model, mem_proj, _tok, _cfg)
