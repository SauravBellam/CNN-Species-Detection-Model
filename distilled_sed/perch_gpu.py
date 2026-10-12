"""Perch v2 on the GPU only -- no CPU execution anywhere.

Two backends, picked from the path:
  *.onnx            onnxruntime-gpu with ONLY the CUDA execution provider and CPU
                    fallback disabled (session.disable_cpu_ep_fallback). If any node
                    of the graph has no CUDA kernel, building the session FAILS
                    instead of quietly running that node on the CPU.
  SavedModel dir    Google's GPU Perch v2 TensorFlow SavedModel (Kaggle Models:
                    google/bird-vocalization-classifier, TensorFlow 2, "perch_v2" --
                    NOT the "perch_v2_cpu" variant), run under tf.device("/GPU:0");
                    fails if TensorFlow sees no GPU.

Both return per-window logits (Perch's ~15k classes), the 1536-d embedding and,
when available, per-frame (spatial) embeddings averaged over frequency.
"""
from pathlib import Path

import numpy as np


class PerchGPU:
    def __init__(self, path, batch=128):
        self.path, self.batch = Path(path), batch
        if self.path.is_dir():
            self._init_tf()
        else:
            self._init_onnx()
        print(f"Perch backend: {self.backend} (GPU only)", flush=True)

    # ---------------- ONNX Runtime, CUDA only
    def _init_onnx(self):
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("onnxruntime has no CUDA provider: install onnxruntime-gpu "
                               "built for the VM's CUDA version")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        try:
            self.sess = ort.InferenceSession(str(self.path), so, providers=["CUDAExecutionProvider"])
        except Exception as e:
            raise RuntimeError(
                f"Part of {self.path.name} cannot run on the GPU with onnxruntime ({e}).\n"
                "Use Google's GPU Perch v2 TensorFlow SavedModel instead: pass its directory "
                "as the Perch model path (see distilled_sed/perch_gpu.py).") from e
        self.inp = self.sess.get_inputs()[0].name
        names = [o.name for o in self.sess.get_outputs()]
        assert "label" in names and "embedding" in names, f"Perch outputs: {names}"
        self.out = {n: i for i, n in enumerate(names)}
        self.spatial_key = next((n for n in names if "spatial" in n), None)
        self.backend = "onnxruntime CUDAExecutionProvider"

    def _run_onnx(self, x):
        o = self.sess.run(None, {self.inp: x})
        return (o[self.out["label"]], o[self.out["embedding"]],
                o[self.out[self.spatial_key]] if self.spatial_key else None)

    # ---------------- TensorFlow SavedModel on GPU
    def _init_tf(self):
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            raise RuntimeError("TensorFlow sees no GPU: install tensorflow[and-cuda]")
        for g in gpus:  # don't grab all GPU memory
            tf.config.experimental.set_memory_growth(g, True)
        self.tf = tf
        with tf.device("/GPU:0"):
            self.fn = tf.saved_model.load(str(self.path)).signatures["serving_default"]
        keys = list(self.fn.structured_outputs)
        assert "label" in keys and "embedding" in keys, f"Perch outputs: {keys}"
        self.spatial_key = next((k for k in keys if "spatial" in k), None)
        self.backend = f"TensorFlow SavedModel on {gpus[0].name}"

    def _run_tf(self, x):
        with self.tf.device("/GPU:0"):
            o = self.fn(inputs=self.tf.constant(x))
        return (o["label"].numpy(), o["embedding"].numpy(),
                o[self.spatial_key].numpy() if self.spatial_key else None)

    # ---------------- shared
    def __call__(self, x):
        """x (N, 160000) float32 -> logits (N, L), emb (N, 1536), spatial (N, T, 1536) or None."""
        run = self._run_tf if self.path.is_dir() else self._run_onnx
        L, E, S = [], [], []
        for s in range(0, len(x), self.batch):
            lg, em, sp = run(np.ascontiguousarray(x[s:s + self.batch], dtype=np.float32))
            L.append(lg); E.append(em)
            if sp is not None:
                S.append(sp.mean(axis=2) if sp.ndim == 4 else sp)  # (N, T, F, D) -> (N, T, D)
        cat = lambda a: np.concatenate(a).astype(np.float32)
        return cat(L), cat(E), (cat(S) if S else None)
