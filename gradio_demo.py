"""
Gradio Sketch-to-Face Demo (Canny + XDoG) on Modal
==================================================
Draw white edges on a black canvas and generate a photorealistic face using
either the Canny-edge or XDoG-edge Flow Matching model. Switch between them
in the UI.

Deploy:
    modal serve gradio_demo_combined.py   # dev mode (live reload)
    modal deploy gradio_demo_combined.py   # persistent deployment

URL: https://{workspace}--sketch-to-face-combined-ui.modal.run
"""

import modal

# ?????????????????????????????????????????????????????????????????????????????
# Modal App + Image
# ?????????????????????????????????????????????????????????????????????????????

app = modal.App("sketch-to-face-combined")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "pillow>=9.0.0",
        "pyyaml>=6.0",
        "einops>=0.6.0",
        "tqdm>=4.64.0",
        "scipy>=1.9.0",
        "opencv-python-headless>=4.0.0",
        "gradio>=4.0.0",
        "fastapi[standard]>=0.100.0",
        "slowapi>=0.1.9",
        "lpips",
    )
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

volume = modal.Volume.from_name("cmu-10799-diffusion-data", create_if_missing=True)
DATA_DIR = "/data"

# ?? Checkpoints directories on the volume (under DATA_DIR). Latest flow_matching_*.pt is chosen from each.
BASELINE_CHECKPOINTS_DIR = "logs/edge_canny_flow_matching_modal_2gpu/flow_matching_20260217_225006/checkpoints"
BASELINE_BGAUG_CHECKPOINTS_DIR = "logs/hw4_noadapter_canny_mix_eagf_geomaug_precomputedbg_modal_4gpu_fastfull/flow_matching_20260226_111048/checkpoints"
CANNY_CHECKPOINTS_DIR = "logs/hw4_edge_adapter_canny_mix_eagf_geomaug_precomputedbg_modal_2gpu_canny10_no23/flow_matching_20260225_031235/checkpoints"
XDOG_CHECKPOINTS_DIR = "logs/hw4_edge_xdog028_eagf_modal_2gpu/flow_matching_20260221_152822/checkpoints"
XDOG_BGAUG_CHECKPOINTS_DIR = "logs/hw4_edge_adapter_xdog028_eagf_geomaug_precomputedbg_modal_2gpu/flow_matching_20260226_091124/checkpoints"


def find_latest_checkpoint(
    data_dir: str = DATA_DIR,
    subdir_contains: str | None = None,
    subdir_excludes: str | None = None,
) -> str | None:
    """
    Scan the volume for the highest-step flow_matching checkpoint.
    If subdir_contains is set, only paths under dirs containing that string are considered.
    If subdir_excludes is set, paths containing that string are skipped.
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
            if subdir_contains is not None and subdir_contains not in dirpath:
                continue
            if subdir_excludes is not None and subdir_excludes in dirpath:
                continue
            for fname in filenames:
                m = step_re.match(fname)
                if m:
                    step = int(m.group(1))
                    candidates.append((step, os.path.join(dirpath, fname)))
                elif fname == "flow_matching_final.pt":
                    candidates.append((int(1e12), os.path.join(dirpath, fname)))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_step, best_path = candidates[0]
    print(f"[demo] Found {len(candidates)} checkpoint(s). Using: {best_path} (step {best_step:,})")
    return best_path


def find_latest_in_checkpoints_dir(
    data_dir: str,
    checkpoints_dir: str,
) -> str | None:
    """
    Return the path to the latest flow_matching_*.pt in a single checkpoints directory.
    Returns None if the directory does not exist or contains no matching files.
    """
    import os
    import re

    full_dir = os.path.join(data_dir, checkpoints_dir)
    if not os.path.isdir(full_dir):
        return None

    step_re = re.compile(r"flow_matching_(\d+)\.pt$")
    candidates: list[tuple[int, str]] = []
    for fname in os.listdir(full_dir):
        m = step_re.match(fname)
        if m:
            step = int(m.group(1))
            candidates.append((step, os.path.join(full_dir, fname)))
        elif fname == "flow_matching_final.pt":
            candidates.append((int(1e12), os.path.join(full_dir, fname)))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_step, best_path = candidates[0]
    print(f"[demo] Latest in {checkpoints_dir}: {best_path} (step {best_step:,})")
    return best_path


def resolve_checkpoint(
    pinned: str | None,
    data_dir: str,
    subdir_contains: str | None = None,
    subdir_excludes: str | None = None,
) -> str | None:
    """Resolve checkpoint: use pinned path if it exists, else discovery (optionally filtered)."""
    import os
    if pinned:
        path = os.path.join(data_dir, pinned)
        if os.path.isfile(path):
            return path
        print(f"[demo] Pinned checkpoint not found ({path}), trying discovery.")
    return find_latest_checkpoint(
        data_dir,
        subdir_contains=subdir_contains,
        subdir_excludes=subdir_excludes,
    )


def load_model(checkpoint_path: str, device):
    """Load model + EMA from a checkpoint; return (method, image_shape, default_steps)."""
    import sys
    sys.path.insert(0, "/root")

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
    ema.apply_shadow()
    method.eval_mode()

    data_cfg = config["data"]
    image_shape = (data_cfg["channels"], data_cfg["image_size"], data_cfg["image_size"])
    default_steps = config.get("sampling", {}).get("num_steps", 50)
    return method, image_shape, default_steps


# ?????????????????????????????????????????????????????????????????????????????
# Preprocessing: Canny vs XDoG
# ?????????????????????????????????????????????????????????????????????????????

def preprocess_sketch_canny(sketch_pil, image_size: int):
    """
    Canny: white-on-black canvas as-is (black=-1, white=+1). Blank = black.
    """
    import numpy as np
    import torch
    from torchvision import transforms
    from PIL import Image

    if sketch_pil is None:
        blank = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        sketch_pil = Image.fromarray(blank)

    sketch_rgb = sketch_pil.convert("RGB")
    sketch_resized = sketch_rgb.resize((image_size, image_size), Image.BILINEAR)
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return to_tensor(sketch_resized).unsqueeze(0)


def preprocess_sketch_xdog(sketch_pil, image_size: int):
    """
    XDoG model: pass the user's white-on-black sketch through directly.

    The model was trained on XDoG edges which are white edges on a black
    background ? identical in format to what the user draws. No filter is
    applied here; we just resize and normalize.
    """
    import numpy as np
    import torch
    from torchvision import transforms
    from PIL import Image

    if sketch_pil is None:
        blank = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        sketch_pil = Image.fromarray(blank)

    sketch_rgb = sketch_pil.convert("RGB")
    sketch_resized = sketch_rgb.resize((image_size, image_size), Image.BILINEAR)
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return to_tensor(sketch_resized).unsqueeze(0)


# Map UI label -> preprocess_fn. Model key is from EDGE_MODEL_DISPLAY_TO_KEY.
PREPROCESSORS = {
    "Baseline": preprocess_sketch_canny,
    "baseline (+bg +aug)": preprocess_sketch_canny,
    "Best Canny": preprocess_sketch_canny,
    "XDoG": preprocess_sketch_xdog,
    "xdog (+bg+aug)": preprocess_sketch_xdog,
}
# UI display name -> key in models dict
EDGE_MODEL_DISPLAY_TO_KEY = {
    "Baseline": "baseline",
    "baseline (+bg +aug)": "baseline_bgaug",
    "Best Canny": "canny",
    "XDoG": "xdog",
    "xdog (+bg+aug)": "xdog_bgaug",
}


# ?????????????????????????????????????????????????????????????????????????????
# Gradio UI
# ?????????????????????????????????????????????????????????????????????????????

def build_demo(models: dict, default_steps: int):
    """
    Build combined demo.
    Only keys for available models are present.
    """
    import gradio as gr
    import torch
    import numpy as np
    from PIL import Image
    from PIL import Image as PILImage
    from src.data import unnormalize

    choices = [
        name
        for name in ("Baseline", "baseline (+bg +aug)", "Best Canny", "XDoG", "xdog (+bg+aug)")
        if EDGE_MODEL_DISPLAY_TO_KEY[name] in models
    ]
    first_choice = choices[0] if choices else "Baseline"

    def _default_cfg_for_model(edge_model: str) -> float:
        key = EDGE_MODEL_DISPLAY_TO_KEY.get(edge_model, edge_model.lower())
        if key not in models:
            key = EDGE_MODEL_DISPLAY_TO_KEY.get(first_choice, first_choice.lower())
        method, _, _ = models[key]
        return float(getattr(method, "cfg_guidance_scale", 1.0))

    def _sync_cfg_slider(edge_model: str):
        return gr.update(value=_default_cfg_for_model(edge_model))

    def generate(
        sketch_editor_value,
        edge_model: str,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ):
        key = EDGE_MODEL_DISPLAY_TO_KEY.get(edge_model, edge_model.lower())
        if key not in models:
            key = EDGE_MODEL_DISPLAY_TO_KEY.get(first_choice, first_choice.lower())
        method, image_shape, _ = models[key]
        preprocess_fn = PREPROCESSORS[edge_model] if edge_model in PREPROCESSORS else PREPROCESSORS[first_choice]

        sketch_pil = None
        if sketch_editor_value is not None:
            if isinstance(sketch_editor_value, dict):
                sketch_pil = sketch_editor_value.get("composite")
            elif hasattr(sketch_editor_value, "convert"):
                sketch_pil = sketch_editor_value

        if seed >= 0:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed(int(seed))

        channels, h, w = image_shape
        device_obj = next(iter(method.model.parameters())).device
        cond = preprocess_fn(sketch_pil, h).to(device_obj)

        with torch.no_grad():
            samples = method.sample(
                batch_size=1,
                image_shape=image_shape,
                num_steps=int(num_steps),
                condition=cond,
                guidance_scale=float(guidance_scale),
            )

        out = unnormalize(samples)
        out_np = (out[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_out = Image.fromarray(out_np)
        pil_out = pil_out.resize((512, 512), Image.Resampling.LANCZOS)
        return pil_out

    black_bg = PILImage.new("RGB", (512, 512), (0, 0, 0))

    def clear_sketch():
        """Reset sketchpad to black background (not white)."""
        reset = PILImage.new("RGB", (512, 512), (0, 0, 0))
        return gr.update(value={
            "background": reset,
            "layers": [],
            "composite": reset,
        })

    def apply_max_perceptual_qual():
        """Preset: lowest CFG and sampling steps for max perceptual quality."""
        return gr.update(value=10), gr.update(value=1.0)

    def save_sketch_64x64(sketch_editor_value):
        """Export the current sketch as 64x64 PNG so the browser can download it locally."""
        import tempfile
        from datetime import datetime
        sketch_pil = None
        if sketch_editor_value is not None:
            if isinstance(sketch_editor_value, dict):
                sketch_pil = sketch_editor_value.get("composite")
            elif hasattr(sketch_editor_value, "convert"):
                sketch_pil = sketch_editor_value
        if sketch_pil is None:
            sketch_pil = PILImage.new("RGB", (512, 512), (0, 0, 0))
        sketch_64 = sketch_pil.convert("RGB").resize((64, 64), PILImage.BILINEAR)
        filename = f"my_sketch_64x64_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            sketch_64.save(f.name)
            return (f.name, filename)

    CANVAS_SIZE = 512

    # Mobile: prevent touch from being used for scroll so canvas gets draw events
    _css = """
    #sketch-pad { touch-action: none; }
    #sketch-pad canvas { touch-action: none; user-select: none; -webkit-tap-highlight-color: transparent; }
    """
    # When in-canvas trash is clicked, trigger our Clear button so canvas resets to black (not white)
    # On iPad: close brush-size popover when tapping outside or on the same button (tap-to-close fix)
    _js = """
    (function() {
      function wireTrashToClear() {
        var sketch = document.getElementById("sketch-pad");
        var clearBtn = document.getElementById("clear-drawing-btn");
        if (!sketch || !clearBtn) return false;
        var buttons = sketch.querySelectorAll("button");
        for (var i = 0; i < buttons.length; i++) {
          var btn = buttons[i];
          var label = (btn.getAttribute("aria-label") || btn.getAttribute("title") || "").toLowerCase();
          var testId = (btn.getAttribute("data-testid") || "").toLowerCase();
          var isTrash = /clear|delete|trash|reset/.test(label) || /clear|delete|trash|reset/.test(testId);
          if (isTrash && !btn.dataset.wiredToClear) {
            btn.dataset.wiredToClear = "1";
            btn.addEventListener("click", function() {
              setTimeout(function() { clearBtn.click(); }, 50);
            });
          }
        }
        return true;
      }
      function closeBrushPopoverIfOpen(e) {
        var sketch = document.getElementById("sketch-pad");
        if (!sketch) return;
        var target = e.target;
        var expanded = sketch.querySelector('[aria-expanded="true"]');
        if (!expanded) {
          var openPopover = sketch.querySelector('[data-state="open"], .open, [aria-hidden="false"]');
          if (openPopover && openPopover.closest && !openPopover.contains(target)) {
            var trigger = sketch.querySelector('[aria-haspopup="true"], [data-popover-trigger]');
            if (trigger) trigger.click();
          }
          return;
        }
        var popoverId = expanded.getAttribute("aria-controls");
        var popover = popoverId ? document.getElementById(popoverId) : expanded.nextElementSibling;
        var insidePopover = popover && popover.contains && popover.contains(target);
        if (insidePopover) return;
        var onTrigger = expanded.contains && expanded.contains(target);
        if (onTrigger) {
          e.preventDefault();
          e.stopPropagation();
        }
        expanded.click();
      }
      function setupBrushPopoverClose() {
        var sketch = document.getElementById("sketch-pad");
        if (!sketch || sketch.dataset.brushCloseWired) return;
        sketch.dataset.brushCloseWired = "1";
        ["touchstart", "touchend", "mousedown"].forEach(function(ev) {
          document.addEventListener(ev, closeBrushPopoverIfOpen, true);
        });
      }
      if (document.readyState === "complete") {
        setTimeout(wireTrashToClear, 500);
        setTimeout(setupBrushPopoverClose, 600);
      } else {
        window.addEventListener("load", function() {
          setTimeout(wireTrashToClear, 500);
          setTimeout(setupBrushPopoverClose, 600);
        });
      }
      var check = setInterval(function() {
        if (wireTrashToClear()) clearInterval(check);
      }, 1000);
      setTimeout(function() { clearInterval(check); }, 10000);
      setInterval(setupBrushPopoverClose, 2000);
    })();
    """

    with gr.Blocks(
        title="Sketch ? Face (Canny / XDoG)",
        fill_width=True,
        head="<meta name='viewport' content='width=device-width, initial-scale=1'>",
        css=_css,
        js=_js,
    ) as demo:
        gr.Markdown(
            """
            # Sketch ? Face
            Draw white edge strokes on the black canvas, then click **Generate**.
            """
        )

        # Compact top bar: controls in one row
        with gr.Row(variant="compact"):
            edge_radio = gr.Radio(
                label="Edge model",
                choices=choices,
                value=first_choice,
            )
            num_steps_slider = gr.Slider(
                minimum=10,
                maximum=100,
                step=5,
                value=10,
                label="Steps",
            )
            guidance_scale_slider = gr.Slider(
                minimum=1.0,
                maximum=10.0,
                step=0.5,
                value=1.0,
                label="CFG",
            )
            preset_btn = gr.Button("Max perceptual qual", variant="secondary")
            clear_btn = gr.Button("Clear drawing", variant="secondary", elem_id="clear-drawing-btn")
            save_sketch_btn = gr.DownloadButton("Save my sketch (64?64)", variant="secondary")
            generate_btn = gr.Button("Generate face", variant="primary", size="lg")
        with gr.Accordion("Advanced", open=False):
            seed_slider = gr.Slider(
                minimum=-1,
                maximum=9999,
                step=1,
                value=42,
                label="Seed  (-1 = random)",
            )

        # Main content: sketch | output
        with gr.Row():
            sketch_input = gr.Sketchpad(
                label="Draw your face sketch (white on black)",
                canvas_size=(512, 512),
                height=CANVAS_SIZE,
                width=CANVAS_SIZE,
                type="pil",
                image_mode="RGB",
                brush=gr.Brush(
                    colors=["#FFFFFF"],
                    color_mode="fixed",
                    default_color="#FFFFFF",
                    default_size=2,
                ),
                value={
                    "background": black_bg,
                    "layers": [],
                    "composite": black_bg,
                },
                elem_id="sketch-pad",
            )
            output_image = gr.Image(
                label="Generated face",
                type="pil",
                height=CANVAS_SIZE,
                width=CANVAS_SIZE,
            )

        edge_radio.change(
            fn=_sync_cfg_slider,
            inputs=edge_radio,
            outputs=guidance_scale_slider,
        )
        preset_btn.click(
            fn=apply_max_perceptual_qual,
            outputs=[num_steps_slider, guidance_scale_slider],
            queue=False,
        )
        clear_btn.click(fn=clear_sketch, outputs=sketch_input)
        save_sketch_btn.click(
            fn=save_sketch_64x64,
            inputs=sketch_input,
            outputs=save_sketch_btn,
        )
        generate_btn.click(
            fn=generate,
            inputs=[sketch_input, edge_radio, num_steps_slider, guidance_scale_slider, seed_slider],
            outputs=output_image,
        )

        with gr.Accordion("Details", open=False):
            gr.Markdown(
                """
                **Architecture:** UNet (6-ch input = 3-ch noisy image + 3-ch edge map)  |
                **Method:** Flow Matching, Euler steps  |  **Models:** Baseline + Best Canny + XDoG, CelebA 64?64
                """
            )

    return demo


def make_fastapi_app() -> "FastAPI":
    from fastapi import FastAPI
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware

    limiter = Limiter(key_func=get_remote_address, default_limits=["20/minute"])
    fast_app = FastAPI()
    fast_app.state.limiter = limiter
    fast_app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    fast_app.add_middleware(SlowAPIMiddleware)

    @fast_app.get("/health")
    async def health():
        return {"status": "ok"}

    return fast_app


@app.local_entrypoint()
def main():
    print("To run the combined Canny/XDoG sketch-to-face demo, use:")
    print("  modal serve gradio_demo_combined.py")
    print("Then open the URL printed by Modal in your browser.")
    print("Do not use 'modal run' ? use 'modal serve'.")


@app.function(
    image=image,
    gpu="L40S",
    volumes={DATA_DIR: volume},
    max_containers=1,
    timeout=600,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def ui():
    """Load baseline/Canny/XDoG variants (when available) and serve the combined Gradio demo."""
    import sys
    sys.path.insert(0, "/root")

    import torch
    import gradio as gr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[demo] Device: {device}")

    # Resolve and load each checkpoint (latest in known dirs, else discovery)
    ckpt_baseline = find_latest_in_checkpoints_dir(DATA_DIR, BASELINE_CHECKPOINTS_DIR) or find_latest_checkpoint(
        DATA_DIR, subdir_contains="edge_canny_flow_matching_modal_2gpu"
    )
    ckpt_baseline_bgaug = find_latest_in_checkpoints_dir(DATA_DIR, BASELINE_BGAUG_CHECKPOINTS_DIR) or find_latest_checkpoint(
        DATA_DIR, subdir_contains="hw4_noadapter_canny_mix_eagf_geomaug_precomputedbg_modal_4gpu_fastfull"
    )
    ckpt_canny = find_latest_in_checkpoints_dir(DATA_DIR, CANNY_CHECKPOINTS_DIR) or find_latest_checkpoint(
        DATA_DIR, subdir_contains="edge_canny"
    )
    ckpt_xdog = find_latest_in_checkpoints_dir(DATA_DIR, XDOG_CHECKPOINTS_DIR) or find_latest_checkpoint(
        DATA_DIR,
        subdir_contains="edge_xdog",
        subdir_excludes="edge_canny",
    )
    ckpt_xdog_bgaug = find_latest_in_checkpoints_dir(DATA_DIR, XDOG_BGAUG_CHECKPOINTS_DIR) or find_latest_checkpoint(
        DATA_DIR, subdir_contains="hw4_edge_adapter_xdog028_eagf_geomaug_precomputedbg_modal_2gpu"
    )

    if (
        ckpt_baseline is None
        and ckpt_baseline_bgaug is None
        and ckpt_canny is None
        and ckpt_xdog is None
        and ckpt_xdog_bgaug is None
    ):
        with gr.Blocks(title="No checkpoint found") as err_demo:
            gr.Markdown(
                f"""
                # No checkpoint found

                No `flow_matching_*.pt` files were found under `{DATA_DIR}/logs/`
                or `{DATA_DIR}/checkpoints/`.

                Train at least one model, e.g.:
                ```bash
                modal run --detach modal_app.py::main --action train --method flow_matching \\
                    --config configs/edge_canny_flow_matching_modal_2gpu.yaml
                ```
                or `configs/edge_flow_matching_modal_4gpu.yaml` for XDoG.

                Redeploy this demo after checkpoints exist.
                """
            )
        fast_app = make_fastapi_app()
        return gr.mount_gradio_app(fast_app, err_demo, path="/")

    models = {}
    default_steps = 50

    if ckpt_baseline:
        method, image_shape, steps = load_model(ckpt_baseline, device)
        models["baseline"] = (method, image_shape, steps)
        default_steps = steps
        print(f"[demo] Baseline model ready | image_shape={image_shape} | default_steps={steps}")

    if ckpt_baseline_bgaug:
        method, image_shape, steps = load_model(ckpt_baseline_bgaug, device)
        models["baseline_bgaug"] = (method, image_shape, steps)
        if "baseline" not in models:
            default_steps = steps
        print(f"[demo] baseline (+bg +aug) model ready | image_shape={image_shape} | default_steps={steps}")

    if ckpt_canny:
        method, image_shape, steps = load_model(ckpt_canny, device)
        models["canny"] = (method, image_shape, steps)
        if "baseline" not in models and "baseline_bgaug" not in models:
            default_steps = steps
        print(f"[demo] Canny model ready | image_shape={image_shape} | default_steps={steps}")

    if ckpt_xdog:
        method, image_shape, steps = load_model(ckpt_xdog, device)
        models["xdog"] = (method, image_shape, steps)
        if "baseline" not in models and "baseline_bgaug" not in models and "canny" not in models:
            default_steps = steps
        print(f"[demo] XDoG model ready | image_shape={image_shape} | default_steps={steps}")

    if ckpt_xdog_bgaug:
        method, image_shape, steps = load_model(ckpt_xdog_bgaug, device)
        models["xdog_bgaug"] = (method, image_shape, steps)
        if "baseline" not in models and "baseline_bgaug" not in models and "canny" not in models and "xdog" not in models:
            default_steps = steps
        print(f"[demo] xdog (+bg+aug) model ready | image_shape={image_shape} | default_steps={steps}")

    demo = build_demo(models, default_steps)
    fast_app = make_fastapi_app()
    return gr.mount_gradio_app(fast_app, demo, path="/")

