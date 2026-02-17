"""
Gradio Sketch-to-Face Demo on Modal
====================================
Draw an anti-aliased sketch in the browser and generate a photorealistic face
using the trained edge-conditioned Flow Matching model.

Deploy:
    modal serve gradio_demo.py        # dev mode (live reload)
    modal deploy gradio_demo.py       # persistent deployment

The URL will look like:
    https://{workspace}--sketch-to-face-ui.modal.run
"""

import modal

# ─────────────────────────────────────────────────────────────────────────────
# Modal App + Image
# ─────────────────────────────────────────────────────────────────────────────

app = modal.App("sketch-to-face")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # ML stack
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "pillow>=9.0.0",
        "pyyaml>=6.0",
        "einops>=0.6.0",
        "tqdm>=4.64.0",
        "scipy>=1.9.0",
        "opencv-python-headless>=4.0.0",
        # Web stack
        "gradio>=4.0.0",
        "fastapi[standard]>=0.100.0",
        "slowapi>=0.1.9",       # per-IP rate limiting middleware
    )
    # Copy the whole repo into the image (same pattern as modal_app.py)
    .add_local_dir(
        ".",
        "/root",
        ignore=[
            ".git", ".venv*", "venv", "__pycache__",
            "logs", "checkpoints", "*.md", "docs",
            "environments", "notebooks",
        ],
    )
)

# Same persistent volume that stores checkpoints and logs from training
volume = modal.Volume.from_name("cmu-10799-diffusion-data", create_if_missing=True)

DATA_DIR = "/data"  # volume mount point inside the container

# ── Pinned checkpoint ─────────────────────────────────────────────────────────
# Set to a relative path under DATA_DIR to use a specific checkpoint, or None
# to fall back to automatic discovery (highest-step checkpoint found on volume).
PINNED_CHECKPOINT = (
    "logs/edge_flow_matching_modal_4gpu"
    "/flow_matching_20260217_192112"
    "/checkpoints/flow_matching_0015000.pt"
)

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint Discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(data_dir: str = DATA_DIR) -> str | None:
    """
    Scan the volume for the highest-step flow_matching checkpoint.

    Search paths:
      - /data/logs/*/checkpoints/flow_matching_*.pt   (timestamped run dirs)
      - /data/checkpoints/*/flow_matching_*.pt         (legacy layout)

    Returns the absolute path of the best checkpoint, or None if nothing found.
    """
    import os
    import re

    candidates: list[tuple[int, str]] = []

    search_roots = [
        os.path.join(data_dir, "logs"),
        os.path.join(data_dir, "checkpoints"),
    ]

    step_re = re.compile(r"flow_matching_(\d+)\.pt$")

    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for fname in filenames:
                m = step_re.match(fname)
                if m:
                    step = int(m.group(1))
                    candidates.append((step, os.path.join(dirpath, fname)))
                elif fname == "flow_matching_final.pt":
                    # treat final as step = inf so it's always preferred
                    candidates.append((int(1e12), os.path.join(dirpath, fname)))

    if not candidates:
        return None

    # Return path with the highest step
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_step, best_path = candidates[0]
    print(f"[demo] Found {len(candidates)} checkpoint(s). Using: {best_path} (step {best_step:,})")
    return best_path


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (adapted from sample.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device):
    """Load model + EMA from a checkpoint; return (method, image_shape, default_steps)."""
    import sys
    sys.path.insert(0, "/root")  # ensure src/ is importable inside Modal container

    import torch
    from src.models import create_model_from_config
    from src.methods import FlowMatching
    from src.utils import EMA

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = create_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model"])

    ema = EMA(model, decay=config["training"]["ema_decay"])
    ema.load_state_dict(checkpoint["ema"])

    method = FlowMatching.from_config(model, config, device)
    ema.apply_shadow()   # use EMA weights for inference
    method.eval_mode()

    data_cfg = config["data"]
    image_shape = (data_cfg["channels"], data_cfg["image_size"], data_cfg["image_size"])
    default_steps = config.get("sampling", {}).get("num_steps", 50)

    return method, image_shape, default_steps


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing: user sketch → normalized conditioning tensor
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_sketch(sketch_pil, image_size: int):
    """
    Convert the Sketchpad PIL image into a conditioning tensor.

    The Sketchpad returns a composite PIL image with:
      - white background (255, 255, 255)
      - dark user strokes (near-black, anti-aliased)

    XDoG edges used during training are also dark-on-white, so we just
    resize to 64×64 and apply the same normalization used at training time.
    """
    import numpy as np
    import torch
    from torchvision import transforms
    from PIL import Image

    if sketch_pil is None:
        # Blank white canvas — no condition signal
        blank = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
        sketch_pil = Image.fromarray(blank)

    # Ensure RGB (ImageEditor outputs can be RGBA)
    sketch_rgb = sketch_pil.convert("RGB")
    sketch_resized = sketch_rgb.resize((image_size, image_size), Image.BILINEAR)

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return to_tensor(sketch_resized).unsqueeze(0)  # (1, 3, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

def build_demo(method, image_shape, default_steps: int):
    """Build and return the Gradio Blocks demo."""
    import gradio as gr
    import torch
    import numpy as np
    from PIL import Image
    from src.data import unnormalize

    channels, h, w = image_shape
    device_obj = next(iter(method.model.parameters())).device

    def generate(sketch_editor_value, num_steps: int, seed: int):
        """
        Run one inference pass given the Sketchpad output.

        In Gradio 4.x, gr.Sketchpad (a.k.a. gr.ImageEditor with sources=())
        returns a dict with keys:  background, layers, composite.
        We use `composite` — the final flattened image with all strokes.
        """
        # Extract the composite (drawn) image from the editor value dict
        sketch_pil = None
        if sketch_editor_value is not None:
            if isinstance(sketch_editor_value, dict):
                sketch_pil = sketch_editor_value.get("composite")
            elif hasattr(sketch_editor_value, "convert"):
                # Already a PIL image (older Gradio versions)
                sketch_pil = sketch_editor_value

        if seed >= 0:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed(int(seed))

        cond = preprocess_sketch(sketch_pil, h).to(device_obj)

        with torch.no_grad():
            samples = method.sample(
                batch_size=1,
                image_shape=image_shape,
                num_steps=int(num_steps),
                condition=cond,
            )  # (1, C, H, W) in [-1, 1]

        # Unnormalize [-1, 1] → [0, 1] → uint8 PIL
        out = unnormalize(samples)
        out_np = (out[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(out_np)

    # ── Layout ────────────────────────────────────────────────────────────────
    with gr.Blocks(title="Sketch → Face") as demo:
        gr.Markdown(
            """
            # Sketch → Face
            Draw a face outline in the canvas on the left, then click **Generate**.

            The model was trained on CelebA 64×64 faces using edge-conditioned **Flow Matching**
            (pixel-space, XDoG sketch concatenation).

            **Drawing tips:**
            - Draw dark strokes on the white canvas — this matches the training edge style.
            - Sketch a rough face: oval outline, two dots for eyes, short lines for nose and mouth.
            - Use the eraser button to fix mistakes.
            - Try different random seeds for variety.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # gr.Sketchpad = gr.ImageEditor(sources=(), brush=Brush(colors=["#000000"], color_mode="fixed"))
                # canvas_size sets the actual pixel resolution; height/width sets display size.
                sketch_input = gr.Sketchpad(
                    label="Draw your face sketch here",
                    canvas_size=(512, 512),
                    height=512,
                    width=512,
                    type="pil",
                    image_mode="RGB",
                )
                with gr.Row():
                    num_steps_slider = gr.Slider(
                        minimum=10,
                        maximum=100,
                        step=5,
                        value=default_steps,
                        label="Sampling steps",
                    )
                    seed_slider = gr.Slider(
                        minimum=-1,
                        maximum=9999,
                        step=1,
                        value=42,
                        label="Seed  (-1 = random)",
                    )
                generate_btn = gr.Button("Generate face", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Generated face (64×64, upscaled for display)",
                    type="pil",
                    height=512,
                    width=512,
                )
                gr.Markdown(
                    "_Model output is 64×64 — the browser scales up for display._"
                )

        generate_btn.click(
            fn=generate,
            inputs=[sketch_input, num_steps_slider, seed_slider],
            outputs=output_image,
        )

        gr.Markdown(
            """
            ---
            **Architecture:** UNet (6-ch input = 3-ch noisy image + 3-ch edge map)  |
            **Method:** Flow Matching, 50 Euler steps  |
            **Training:** 120K iters, 4× L40S GPUs, CelebA 64×64
            """
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Rate-limited FastAPI app factory
# ─────────────────────────────────────────────────────────────────────────────

def make_fastapi_app() -> "FastAPI":
    """
    Build a FastAPI app with per-IP rate limiting via slowapi.

    Limits:
      - /generate endpoint: 10 requests / minute per IP
      - Everything else: unlimited (static assets, Gradio WebSocket, etc.)

    The Gradio inference click fires a POST to /queue/join, which is the
    actual heavy call — that's what we rate-limit via the global limiter.
    All requests are counted; heavy ones are handled by the global default.
    """
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware

    # Key by remote IP address
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=["20/minute"],   # global default: 20 req/min per IP
    )

    fast_app = FastAPI()
    fast_app.state.limiter = limiter
    fast_app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    fast_app.add_middleware(SlowAPIMiddleware)

    # Optional: expose a /health endpoint (not rate-limited by default)
    @fast_app.get("/health")
    async def health():
        return {"status": "ok"}

    return fast_app


# ─────────────────────────────────────────────────────────────────────────────
# Modal Function: serve the Gradio app
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="L40S",
    volumes={DATA_DIR: volume},
    max_containers=1,   # sticky sessions — Gradio requires all requests to same container
    timeout=600,        # 10 min idle timeout before container recycles
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def ui():
    """Load the trained model and serve the Gradio sketch-to-face demo."""
    import sys
    sys.path.insert(0, "/root")

    import torch
    from gradio.routes import mount_gradio_app
    import gradio as gr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[demo] Device: {device}")

    # ── Resolve checkpoint ────────────────────────────────────────────────────
    import os
    if PINNED_CHECKPOINT:
        pinned = os.path.join(DATA_DIR, PINNED_CHECKPOINT)
        if os.path.isfile(pinned):
            print(f"[demo] Using pinned checkpoint: {pinned}")
            ckpt_path = pinned
        else:
            print(f"[demo] Pinned checkpoint not found ({pinned}), falling back to auto-discovery.")
            ckpt_path = find_latest_checkpoint(DATA_DIR)
    else:
        ckpt_path = find_latest_checkpoint(DATA_DIR)

    if ckpt_path is None:
        # No checkpoint yet: serve a friendly error page
        with gr.Blocks(title="No checkpoint found") as err_demo:
            gr.Markdown(
                f"""
                # No checkpoint found

                No `flow_matching_*.pt` files were found under `{DATA_DIR}/logs/`
                or `{DATA_DIR}/checkpoints/`.

                Please train the model first:
                ```bash
                modal run --detach modal_app.py::main \\
                    --action train \\
                    --method flow_matching \\
                    --config configs/edge_flow_matching_modal_4gpu.yaml
                ```

                Once training has saved at least one checkpoint (every 5K steps),
                redeploy this demo.
                """
            )
        return mount_gradio_app(app=make_fastapi_app(), blocks=err_demo, path="/")

    # ── Load model ────────────────────────────────────────────────────────────
    method, image_shape, default_steps = load_model(ckpt_path, device)
    print(f"[demo] Model ready | image_shape={image_shape} | default_steps={default_steps}")

    # ── Build and serve Gradio UI ─────────────────────────────────────────────
    demo = build_demo(method, image_shape, default_steps)
    return mount_gradio_app(app=make_fastapi_app(), blocks=demo, path="/")
