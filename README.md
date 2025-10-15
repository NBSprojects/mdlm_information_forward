# text8-beta-diffusion

Repo factorisé à partir d'un notebook pour entraîner un débruiteur discret sur **text8**
avec un **masquage Beta par token** piloté par un **Scheduler MLP**.

### Ajouts clefs
- **Logs**: affichage **simultané** de la loss **unweighted** et **weighted**.
- **Précision**: **`bf16_only`** (par défaut `False`) via `PrecisionPolicy`.
- **Sampling**: **Top‑p (nucleus) sampling** optionnel (`top_p < 1`).
- **Fidélité**: implémentation au plus près du notebook original.

## Installation rapide
```bash
pip install -e .
# ou
pip install -r requirements.txt
```

## Entraînement MLP
```bash
python -m scripts.train_scheduler_cli
```

## Entraînement débruiteur
```bash
python -m scripts.train_denoiser_cli --top-p 0.9
python -m scripts.train_denoiser_cli --bf16-only --top-p 0.9
```

## Sampling
```bash
python -m scripts.sample_cli --top-p 0.92 --steps 600 --grid cosine --temperature 0.9
```
