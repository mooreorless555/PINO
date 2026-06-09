"""
RunPod Serverless handler for PINO (Person-Interaction Noise Optimization).

Wraps the proven `tools/infer_opt.py` generation flow:
  - Loads the InterGen model + checkpoint ONCE at container boot (cached across
    requests on a warm worker) — the expensive part.
  - Per request: runs the noise-optimization loop and returns the raw motion tensor.
  - The mp4 render step in generate_one_sample is monkeypatched to a no-op, since the
    API returns numeric motion, not video (also removes the ffmpeg dependency at runtime).

Input  (job["input"]):
  {
    "prompt": "Two people shake hands and walk away.",   # required
    "motion": [[...], ...]   # optional: a previous result tensor to add a 3rd+ person
  }

Output:
  {
    "prompt": ...,
    "shape": [1, 210, 524],          # [batch, frames, people * 262]
    "num_people": 2,
    "feature_dim_per_person": 262,   # InterGen rep: pos66 + rot126 + vel66 + foot4
    "fps": 30,
    "motion": [[...], ...],          # [frames, people*262] nested list
    "format": "InterGen per-person 262-dim, concatenated on the feature axis"
  }
"""
import os
import sys
import glob
import time
import traceback

# Must be set before matplotlib is imported (pulled in via tools.infer_opt) — headless render.
os.environ.setdefault("MPLBACKEND", "Agg")

# The repo root: where this handler lives (the cloned PINO repo).
REPO_DIR = os.environ.get("PINO_DIR", os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

import torch
import runpod
from collections import OrderedDict
from configs import get_config
from tools import infer_opt as pino  # defines LitGenModel, build_models; __main__ is guarded

LITMODEL = None
NFEATS = None


def _resolve_checkpoint(default_path):
    """Find intergen.ckpt: env override, then the config default, then a mounted volume."""
    candidates = [
        os.environ.get("PINO_CHECKPOINT"),
        default_path,                                       # checkpoints/intergen.ckpt (baked image)
        "/runpod-volume/PINO/checkpoints/intergen.ckpt",    # pino_data network volume
        "/runpod-volume/checkpoints/intergen.ckpt",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        "intergen.ckpt not found. Bake it into the image (prepare/download_pretrain_model.sh) "
        "or attach the pino_data network volume. Tried: %s" % [c for c in candidates if c]
    )


def _load_model():
    """Build the InterGen model + LitGenModel once; cache globally."""
    global LITMODEL, NFEATS
    if LITMODEL is not None:
        return LITMODEL

    model_cfg = get_config("configs/model.yaml")
    infer_cfg = get_config("configs/infer.yaml")

    # Point the (frozen) config at wherever the checkpoint actually lives.
    ckpt_path = _resolve_checkpoint(model_cfg.CHECKPOINT)
    model_cfg.defrost()
    model_cfg.CHECKPOINT = ckpt_path
    model_cfg.freeze()

    model = pino.build_models(model_cfg)

    ckpt = torch.load(model_cfg.CHECKPOINT, map_location="cpu", weights_only=False)
    new_state_dict = OrderedDict()
    for k, v in ckpt["state_dict"].items():
        new_state_dict[k.replace("model.", "") if "model." in k else k] = v
    model.load_state_dict(new_state_dict, strict=False)
    print("checkpoint state loaded!", flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA device — PINO is very slow on CPU.", flush=True)

    litmodel = pino.LitGenModel(model, infer_cfg).to(device)
    # API returns numeric motion, not video — skip the mp4 render (and its ffmpeg dependency).
    litmodel.plot_t2m = lambda *a, **k: None

    LITMODEL = litmodel
    NFEATS = int(infer_cfg.OPT.INPUT_DIM)  # 262
    return litmodel


def handler(job):
    try:
        job_input = job.get("input", {}) or {}
        prompt = job_input.get("prompt")
        if not prompt or not isinstance(prompt, str):
            return {"error": "Missing or invalid 'prompt' (expected a non-empty string)."}

        litmodel = _load_model()

        # Optional: extend an existing motion with another person (the N-person path).
        motion_data = None
        prev = job_input.get("motion")
        if prev is not None:
            motion_data = torch.tensor(prev, dtype=torch.float32)
            if motion_data.dim() == 2:
                motion_data = motion_data.unsqueeze(0)

        job_id = str(job.get("id", "local")).replace("/", "_")
        out_dir = os.path.join("/tmp", "pino_" + job_id)
        os.makedirs(out_dir, exist_ok=True)

        # Same name derivation as infer_opt.py's __main__.
        name = prompt.replace(" ", "_").replace(".", "").replace(",", "")[:48]

        t0 = time.time()
        litmodel.generate_one_sample(prompt, name, motion_data=motion_data, output_dir=out_dir)
        elapsed = round(time.time() - t0, 2)

        pts = sorted(glob.glob(os.path.join(out_dir, "*.pt")), key=os.path.getmtime)
        if not pts:
            return {"error": "Generation produced no output tensor."}
        motion = torch.load(pts[-1], map_location="cpu", weights_only=False)
        if isinstance(motion, torch.Tensor):
            motion = motion.float().cpu()

        num_people = int(motion.shape[-1] // NFEATS)
        result = {
            "prompt": prompt,
            "shape": list(motion.shape),
            "num_people": num_people,
            "feature_dim_per_person": NFEATS,
            "fps": 30,
            "generation_seconds": elapsed,
            "motion": motion.squeeze(0).tolist(),
            "format": "InterGen per-person 262-dim (pos66+rot126+vel66+foot4), concatenated",
        }

        for p in pts:
            try:
                os.remove(p)
            except OSError:
                pass
        return result

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()[-3000:]}


if __name__ == "__main__":
    # Warm the model at container boot so the first real request isn't penalized.
    try:
        _load_model()
    except Exception as e:
        print("Model warmup failed (will retry on first request): %s" % e, flush=True)

    runpod.serverless.start({"handler": handler})
