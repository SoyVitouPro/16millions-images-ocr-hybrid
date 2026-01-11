
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

# -----------------------
# Config (hard-coded)
# -----------------------
@dataclass
class CFG:
    CKPT_PATH: str = "./model/best.pt"
    VOCAB_JSON: str = "./model/vocab_char.json"

    IMG_H: int = 32
    IMG_W: int = 640

    ENC_DIM: int = 224
    RNN_HIDDEN: int = 224
    RNN_LAYERS: int = 1
    DROPOUT: float = 0.15

    USE_DECODER: bool = True
    DEC_DIM: int = 224
    DEC_LAYERS: int = 2
    DEC_HEADS: int = 7
    DEC_FF: int = 896
    MAX_DEC_LEN: int = 240

    USE_CTC: bool = True

    USE_LM: bool = True
    LM_DIM: int = 192
    LM_LAYERS: int = 2
    LM_HEADS: int = 6
    LM_FF: int = 768

    BEAM: int = 6
    BEAM_LENP: float = 0.7
    USE_LM_FUSION_EVAL: bool = True
    LM_FUSION_ALPHA: float = 0.30

    EOS_LOGP_BIAS: float = 0.55
    EOS_BIAS_UNTIL_LEN: int = 24

    UNK_TOKEN: str = "<unk>"
    COLLAPSE_WHITESPACE: bool = True


# -----------------------
# Tokenizer
# -----------------------
class CharTokenizer:
    def __init__(self, vocab_json: str, unk_token: str = "<unk>", collapse_whitespace: bool = True):
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

    def decode_dec(self, ids: List[int]) -> str:
        out = []
        for x in ids:
            if x in (self.dec_pad, self.dec_bos, self.dec_eos):
                continue
            y = x - self.dec_offset
            if 0 <= y < self.vocab_size:
                t = self.id_to_token.get(y, self.unk_token)
                out.append("�" if t == self.unk_token else t)
        s = "".join(out)
        if self.collapse_whitespace:
            s = re.sub(r"\s+", " ", s).strip()
        return s


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
# Model
# -----------------------
class DWResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: Tuple[int, int], dropout: float):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.bn2 = nn.BatchNorm2d(cout)
        self.drop = nn.Dropout2d(dropout)
        self.skip = None
        if stride != (1, 1) or cin != cout:
            self.skip = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, 0, bias=False),
                nn.BatchNorm2d(cout),
            )

    def forward(self, x):
        h = self.dw(x)
        h = self.bn1(h)
        h = F.silu(h, inplace=True)
        h = self.pw(h)
        h = self.bn2(h)
        h = F.silu(h, inplace=True)
        h = self.drop(h)
        s = x if self.skip is None else self.skip(x)
        return h + s


class HybridContextOCR(nn.Module):
    def __init__(self, cfg: CFG, tok: CharTokenizer):
        super().__init__()
        d = cfg.DROPOUT
        self.cfg = cfg
        self.tok = tok

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        self.cnn = nn.Sequential(
            DWResBlock(32, 64, (2, 2), d),
            DWResBlock(64, 128, (2, 1), d),
            DWResBlock(128, cfg.ENC_DIM, (2, 1), d),
            DWResBlock(cfg.ENC_DIM, cfg.ENC_DIM, (1, 1), d),
        )

        self.pre_ln = nn.LayerNorm(cfg.ENC_DIM)

        self.rnn = nn.GRU(
            input_size=cfg.ENC_DIM,
            hidden_size=cfg.RNN_HIDDEN,
            num_layers=cfg.RNN_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=d if cfg.RNN_LAYERS > 1 else 0.0,
        )
        enc_out = cfg.RNN_HIDDEN * 2

        self.use_ctc = cfg.USE_CTC
        if self.use_ctc:
            self.ctc_head = nn.Sequential(
                nn.LayerNorm(enc_out),
                nn.Dropout(d),
                nn.Linear(enc_out, tok.ctc_classes),
            )

        self.use_decoder = cfg.USE_DECODER
        if self.use_decoder:
            self.mem_proj = nn.Linear(enc_out, cfg.DEC_DIM, bias=False)
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
        x = self.stem(imgs)
        x = self.cnn(x)
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)
        x = self.pre_ln(x)
        x, _ = self.rnn(x)
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
def beam_decode_one(model: HybridContextOCR, mem_proj_1: torch.Tensor, tok: CharTokenizer, cfg: CFG) -> str:
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
                logp[tok.dec_eos] = logp[tok.dec_eos] - (cfg.EOS_LOGP_BIAS * 1.5)
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
    # grayscale -> float32 [0,1] -> normalize to [-1,1]
    g = img.convert("L")
    arr = np.array(g, dtype=np.float32) / 255.0  # H,W
    arr = (arr - 0.5) / 0.5                      # normalize
    t = torch.from_numpy(arr).unsqueeze(0)       # 1,H,W
    return t


# -----------------------
# Global init (loads once)
# -----------------------
_cfg = CFG()
_device = "cuda" if torch.cuda.is_available() else "cpu"

_tok = CharTokenizer(_cfg.VOCAB_JSON, unk_token=_cfg.UNK_TOKEN, collapse_whitespace=_cfg.COLLAPSE_WHITESPACE)
_model = HybridContextOCR(_cfg, _tok).to(_device)

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
def ocr_pil(img: Image.Image) -> str:
    img = img.convert("RGB")
    img = _pre(img)
    x = pil_to_norm_tensor_gray(img).unsqueeze(0).to(_device)  # 1,1,H,W
    mem = _model.encode(x)
    mem_proj = _model.mem_proj(mem)
    return beam_decode_one(_model, mem_proj, _tok, _cfg)

