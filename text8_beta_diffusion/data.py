import os, zipfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List

# === Vocabulary (a..z + space) ===
allowed_chars: List[str] = [chr(ord('a')+i) for i in range(26)] + [' ']
stoi = {ch: i for i, ch in enumerate(allowed_chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size: int = len(allowed_chars)
mask_id: int = vocab_size  # <mask> id used in tensors

def get_vocab():
    return allowed_chars, stoi, itos, vocab_size, mask_id

def load_text8(path: str) -> str:
    if not os.path.exists(path):
        print("[WARN] text8 not found at", path, "- using tiny fallback demo text.")
        return "anarchism originated as a term of abuse used by the public "
    with open(path, "rb") as f:
        head = f.read(4)
    if path.endswith(".zip") or head.startswith(b"PK\x03\x04"):
        with zipfile.ZipFile(path) as zf:
            member = next(n for n in zf.namelist() if n.endswith("text8"))
            data = zf.read(member)
        return data.decode("ascii")
    with open(path, "r", encoding="ascii") as f:
        return f.read()

def encode_text(s: str) -> np.ndarray:
    filtered = "".join(ch for ch in s if ch in stoi)
    arr = np.fromiter((stoi.get(ch, stoi[' ']) for ch in filtered), dtype=np.uint16)
    return np.ascontiguousarray(arr)

def decode_ids(ids) -> str:
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    out = []
    for t in ids:
        if t == mask_id:
            out.append("█")
        else:
            out.append(itos.get(int(t), ' '))
    return "".join(out)

class CharDataset(Dataset):
    def __init__(self, ids: np.ndarray, seq_len: int):
        self.ids = ids; self.seq_len = seq_len
    def __len__(self):
        return max(0, len(self.ids) - self.seq_len - 1)
    def __getitem__(self, idx: int):
        x = self.ids[idx:idx+self.seq_len]
        y = self.ids[idx+1:idx+1+self.seq_len]
        return torch.from_numpy(x).long(), torch.from_numpy(y).long()

def make_dataloaders(data_cfg, batch_size: int):
    raw = load_text8(data_cfg.text8_path)
    if data_cfg.lowercase_only:
        raw = raw.lower()
    assert set(raw) <= set("abcdefghijklmnopqrstuvwxyz "), "text8 must be lowercase letters and space"
    enc = encode_text(raw)
    n = int(len(enc) * data_cfg.train_frac)
    train_ids = enc[:n]; val_ids = enc[n:]
    train_ds = CharDataset(train_ids, data_cfg.seq_len)
    val_ds   = CharDataset(val_ids,   data_cfg.seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=data_cfg.num_workers, drop_last=True, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=data_cfg.num_workers, drop_last=True, pin_memory=True)
    return train_loader, val_loader, vocab_size, mask_id
