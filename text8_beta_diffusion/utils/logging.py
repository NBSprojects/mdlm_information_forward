# text8_beta_diffusion/utils/logging.py
from __future__ import annotations
import os, csv
from dataclasses import asdict, is_dataclass
from typing import Any, Dict
import torch.nn as nn

def log(*args, **kwargs):
    # Garder la compat console
    print(*args, **kwargs)

def _flatten_obj(obj) -> Dict[str, Any]:
    if obj is None:
        return {}
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return dict(obj)
    # fallback: aplatir les attributs publics simples
    out = {}
    for k in dir(obj):
        if k.startswith("_"): 
            continue
        v = getattr(obj, k)
        if callable(v):
            continue
        out[k] = v
    return out

def prefix_keys(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}.{k}": v for k, v in d.items()}

class CSVLogger:
    """
    Accumule des lignes en mémoire puis écrit un CSV en fin d'entraînement.
    Les hyperparams (hp.*) sont insérés comme colonnes constantes.
    """
    def __init__(self, path: str, *hyperparam_objs):
        self.path = path
        self.rows = []
        # concatène et préfixe (hp.data.*, hp.diff.*, hp.mlp.*)
        hp = {}
        labels = ["hp.data", "hp.diff", "hp.mlp", "hp.sampling"]
        for label, obj in zip(labels, hyperparam_objs):
            if obj is None: 
                continue
            hp.update(prefix_keys(_flatten_obj(obj), label))
        self.hp = hp

    def log_row(self, row: Dict[str, Any]):
        # copie défensive
        r = dict(row)
        self.rows.append(r)

    def flush(self):
        if not self.rows:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", newline="") as f:
                f.write("step\n")  # csv minimal pour tracer que l'entraînement a tourné
            return

        # colonnes = union des clés + hp.*
        cols = set()
        for r in self.rows:
            cols |= set(r.keys())
        cols |= set(self.hp.keys())

        # forcer 'step' en première colonne si présente
        cols = ["step"] + sorted([c for c in cols if c != "step"])

        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in self.rows:
                w.writerow({**self.hp, **r})


def get_model_summary(model: nn.Module):
    """
    Calcule et affiche le nombre de paramètres et la taille mémoire statique d'un modèle.
    """
    
    # --- 1. Comptage des Paramètres ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print("\n" + "="*40)
    print("       RÉSUMÉ DU MODÈLE")
    print("="*40)
    print(f"Nombre total de paramètres: {total_params:,}")
    print(f"  > Paramètres entraînables:  {trainable_params:,}")
    print(f"  > Paramètres non-entraînables: {non_trainable_params:,}")
    print("-"*40)

    # --- 2. Calcul de la Taille Mémoire Statique ---
    # (Taille des poids et des buffers, sans compter les gradients ou activations)
    
    # Mémoire prise par les paramètres (poids, biais, etc.)
    param_mem_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Mémoire prise par les buffers 
    # (ex: running_mean/var de BatchNorm, ou vos 'rope_cos_f32' ici)
    # model.buffers() inclut ceux enregistrés avec persistent=False
    buffer_mem_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    
    total_mem_bytes = param_mem_bytes + buffer_mem_bytes
    
    print("Empreinte Mémoire Statique (Modèle seul)")
    print(f"  > Mémoire des paramètres: {param_mem_bytes / (1024**2):.3f} MB")
    print(f"  > Mémoire des buffers:    {buffer_mem_bytes / (1024**2):.3f} MB")
    print(f"  > Mémoire totale statique:  {total_mem_bytes / (1024**2):.3f} MB")
    print("-"*40)
    print("Note : La 'mémoire statique' est la place que le modèle prend en VRAM")
    print("AVANT toute passe forward. Elle n'inclut PAS la mémoire dynamique")
    print("utilisée pour les activations ou les gradients pendant l'entraînement.")