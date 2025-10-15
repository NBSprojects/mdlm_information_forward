# text8_beta_diffusion/utils/timing.py
from __future__ import annotations
import torch
from contextlib import contextmanager

class CudaTimers:
    """
    Petit orchestrateur de paires d'événements CUDA.
    Utilisation:
        timers = CudaTimers(enabled=torch.cuda.is_available())
        with timers.phase("forward"):
            logits = model(x)
        ms = timers.elapsed_ms("forward")
    """
    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled and torch.cuda.is_available())
        self._pairs = {}  # name -> [start_event, end_event]

    def _new_event(self):
        return torch.cuda.Event(enable_timing=True)

    def start(self, name: str):
        if not self.enabled: return
        st = self._new_event()
        st.record()
        self._pairs[name] = [st, None]

    def end(self, name: str):
        if not self.enabled: return
        ed = self._new_event()
        ed.record()
        if name not in self._pairs:
            self._pairs[name] = [None, ed]
        else:
            self._pairs[name][1] = ed

    def elapsed_ms(self, name: str) -> float:
        if not self.enabled: return float("nan")
        st, ed = self._pairs.get(name, (None, None))
        if st is None or ed is None:  # pas encore prêt
            return float("nan")
        torch.cuda.synchronize()
        return st.elapsed_time(ed)  # millisecondes

    def elapsed_all_ms(self) -> dict[str, float]:
        return {k: self.elapsed_ms(k) for k in self._pairs.keys()}

    @contextmanager
    def phase(self, name: str):
        self.start(name)
        try:
            yield
        finally:
            self.end(name)
