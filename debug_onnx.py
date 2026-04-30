"""
Compare PyTorch vs ONNX (or ONNX vs ONNX) outputs for the root-level
depth_anything_v2 relative-depth model.

Usage examples:

  # PT vs ONNX with a real image
  python debug_onnx.py \
      --onnx checkpoints/depth_anything_v2_vitl_518_686.onnx \
      --load-from checkpoints/depth_anything_v2_vitl.pth \
      --encoder vitl --image 000005.png --visualise

  # PT vs ONNX with a normalised model (DepthModelWrapper, depth_scale set at export)
  python debug_onnx.py \
      --onnx checkpoints/depth_anything_v2_vitl_dav2_vitl_norm370.onnx \
      --load-from checkpoints/depth_anything_v2_vitl.pth \
      --encoder vitl --depth-scale 370 --image 000005.png

  # ONNX vs ONNX (no PyTorch needed)
  python debug_onnx.py \
      --onnx checkpoints/depth_anything_v2_vitl_518_686.onnx \
      --onnx-ref checkpoints/depth_anything_v2_vitl_dav2_large.onnx

  # Dummy input (no image required)
  python debug_onnx.py \
      --onnx checkpoints/depth_anything_v2_vitl_518_686.onnx \
      --load-from checkpoints/depth_anything_v2_vitl.pth \
      --encoder vitl
"""

import argparse
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import onnxruntime as ort
import onnx

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# ── helpers ────────────────────────────────────────────────────────────────────

def print_separator(title=""):
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print("─" * pad + f" {title} " + "─" * (width - pad - len(title) - 2))
    else:
        print("─" * width)


def print_tensor_preview(arr: np.ndarray, label: str, n: int = 8):
    """Print the first n flat values of a tensor."""
    flat = arr.ravel()
    vals = "  ".join(f"{v:.6f}" for v in flat[:n])
    print(f"  {label} (shape={arr.shape})  first {n} values:")
    print(f"    [{vals}  ...]")


def stats(a: np.ndarray, b: np.ndarray, label_a="PyTorch", label_b="ONNX"):
    a, b = a.astype(np.float64), b.astype(np.float64)
    diff = np.abs(a - b)

    print_separator("Output shape")
    print(f"  {label_a}: {a.shape}")
    print(f"  {label_b}: {b.shape}")

    print_separator("Value range")
    print(f"  {label_a}: min={a.min():.6f}  max={a.max():.6f}  mean={a.mean():.6f}")
    print(f"  {label_b}: min={b.min():.6f}  max={b.max():.6f}  mean={b.mean():.6f}")

    print_separator("Absolute difference  |A − B|")
    print(f"  max   = {diff.max():.6e}")
    print(f"  mean  = {diff.mean():.6e}")
    print(f"  std   = {diff.std():.6e}")
    print(f"  p95   = {np.percentile(diff, 95):.6e}")
    print(f"  p99   = {np.percentile(diff, 99):.6e}")

    mask = np.abs(a) > 1e-6
    if mask.any():
        rel = diff[mask] / np.abs(a[mask])
        print_separator("Relative error  |A−B| / |A|  (where |A|>1e-6)")
        print(f"  max   = {rel.max():.4%}")
        print(f"  mean  = {rel.mean():.4%}")
        print(f"  p95   = {np.percentile(rel, 95):.4%}")

    dot  = (a.ravel() * b.ravel()).sum()
    norm = np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel())
    cos  = dot / norm if norm > 0 else 0.0
    print_separator("Global similarity")
    print(f"  cosine similarity          = {cos:.8f}  (1.0 = identical)")
    print(f"  allclose (atol=1e-3, rtol=1e-3): {np.allclose(a, b, atol=1e-3, rtol=1e-3)}")
    print(f"  allclose (atol=1e-5, rtol=1e-5): {np.allclose(a, b, atol=1e-5, rtol=1e-5)}")
    print_separator()
    return diff


# ── model loading ──────────────────────────────────────────────────────────────

class DepthModelWrapper(nn.Module):
    """Same wrapper used during ONNX export (onnx_export.py)."""
    def __init__(self, model, depth_scale=None):
        super().__init__()
        self.model = model
        self.depth_scale = depth_scale

    def forward(self, x):
        depth = self.model(x)           # (B, H, W)
        depth = depth.unsqueeze(1)      # (B, 1, H, W)
        if self.depth_scale is not None:
            depth = (depth / self.depth_scale).clamp(0.0, 1.0)
        return depth


def load_pytorch_model(args, device):
    sys.path.insert(0, '.')
    from depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    }

    model = DepthAnythingV2(**model_configs[args.encoder])
    try:
        model.load_state_dict(
            torch.load(args.load_from, map_location='cpu', weights_only=False))
    except Exception:
        sd = torch.load(args.load_from, map_location='cpu', weights_only=False)
        clean = {k.replace('module.', ''): v for k, v in sd['model'].items()}
        model.load_state_dict(clean)

    model = model.to(device).eval()

    if args.depth_scale is not None:
        model = DepthModelWrapper(model, depth_scale=args.depth_scale).to(device).eval()
        print(f"  [info] wrapping model with depth_scale={args.depth_scale}")

    return model


def run_pytorch(model, x_np, device):
    t = torch.from_numpy(x_np).to(device)
    with torch.no_grad():
        out = model(t)
    return out.cpu().numpy()


def run_onnx(onnx_path, x_np):
    opts = ort.SessionOptions()
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
    except Exception:
        sess = ort.InferenceSession(onnx_path, sess_options=opts,
                                    providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name
    return sess.run(None, {name: x_np})[0]


# ── preprocessing ──────────────────────────────────────────────────────────────

def build_input_from_image(image_path, input_size):
    """Load a real image and preprocess it to (1, 3, H, W) float32.

    Uses the same ImageNet normalisation as depth_anything_v2.dpt.image2tensor.
    input_size: (H, W)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = input_size
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img  = (img - mean) / std
    return img.transpose(2, 0, 1)[None]   # (1, 3, H, W)


# ── visualisation ──────────────────────────────────────────────────────────────

def visualise(diff, ref_out, onnx_out, label_ref, label_test, save_path=None):
    if not HAS_PLT:
        print("[info] matplotlib not available – skipping visualisation")
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    def _show(ax, data, title):
        im = ax.imshow(np.squeeze(data), cmap='plasma')
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _show(axes[0], ref_out,  f"{label_ref} output")
    _show(axes[1], onnx_out, f"{label_test} output")
    _show(axes[2], diff,     f"Abs diff  |{label_ref} − {label_test}|")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[info] figure saved to {save_path}")
    else:
        plt.show()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare PyTorch vs ONNX outputs for depth_anything_v2 (relative depth)")

    parser.add_argument('--onnx', required=True,
                        help='ONNX model to test')
    parser.add_argument('--onnx-ref', default=None,
                        help='Reference ONNX model (omit to use PyTorch as reference)')

    # PyTorch args (ignored when --onnx-ref is set)
    parser.add_argument('--load-from', default=None,
                        help='PyTorch checkpoint (.pth)')
    parser.add_argument('--encoder', default='vitl',
                        choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--depth-scale', type=float, default=None,
                        help='Wrap model with DepthModelWrapper (same as export). '
                             'Set only if the ONNX was exported with --depth-scale.')

    # Input
    parser.add_argument('--image', default=None,
                        help='Real image path (uses random dummy input if omitted)')
    parser.add_argument('--input-size', type=int, nargs=2, default=[518, 686],
                        metavar=('H', 'W'))
    parser.add_argument('--seed', type=int, default=42)

    # Output
    parser.add_argument('--visualise', action='store_true')
    parser.add_argument('--save-fig', default=None,
                        help='Save comparison figure to this path')
    parser.add_argument('--save-outputs', action='store_true',
                        help='Save PT and ONNX outputs as .npy files')

    args = parser.parse_args()

    # ── validate ONNX ────────────────────────────────────────────────────────
    print_separator("ONNX model check")
    onnx_model = onnx.load(args.onnx)
    onnx.checker.check_model(onnx_model)
    print(f"  {args.onnx}  ✓")
    inp  = onnx_model.graph.input[0]
    out  = onnx_model.graph.output[0]
    print(f"  input : {inp.name}  "
          f"{[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
    print(f"  output: {out.name}  "
          f"{[d.dim_value for d in out.type.tensor_type.shape.dim]}")

    # ── build input ──────────────────────────────────────────────────────────
    H, W = args.input_size
    if args.image:
        print(f"\n[info] loading image: {args.image}  →  ({H}×{W})")
        x_np = build_input_from_image(args.image, (H, W))
    else:
        rng  = np.random.default_rng(args.seed)
        x_np = rng.standard_normal((1, 3, H, W)).astype(np.float32)
        print(f"\n[info] dummy input  shape={x_np.shape}  seed={args.seed}")

    # ── preview input ────────────────────────────────────────────────────────
    print_separator("Input tensor preview")
    print_tensor_preview(x_np, "input")

    # ── run inference ────────────────────────────────────────────────────────
    print_separator("Running inference")

    print(f"  → ONNX: {args.onnx}")
    onnx_out = run_onnx(args.onnx, x_np)
    print(f"     output shape: {onnx_out.shape}")
    print_tensor_preview(onnx_out, "ONNX output")

    if args.onnx_ref:
        print(f"  → ONNX ref: {args.onnx_ref}")
        ref_out = run_onnx(args.onnx_ref, x_np)
        print(f"     output shape: {ref_out.shape}")
        print_tensor_preview(ref_out, "ONNX-ref output")
        label_ref, label_test = "ONNX-ref", "ONNX"
    else:
        if not args.load_from:
            parser.error("--load-from is required when --onnx-ref is not set")
        device = ('cuda' if torch.cuda.is_available()
                  else 'mps' if torch.backends.mps.is_available()
                  else 'cpu')
        print(f"  → PyTorch ({device}): {args.load_from}")
        pt_model = load_pytorch_model(args, device)
        ref_out  = run_pytorch(pt_model, x_np, device)
        print(f"     output shape: {ref_out.shape}")
        print_tensor_preview(ref_out, "PyTorch output")
        label_ref, label_test = "PyTorch", "ONNX"

    # ── align shapes (some exports add an extra dim) ─────────────────────────
    ref_sq  = np.squeeze(ref_out)
    onnx_sq = np.squeeze(onnx_out)
    if ref_sq.shape != onnx_sq.shape:
        print(f"\n[warning] shape mismatch after squeeze: "
              f"{ref_sq.shape} vs {onnx_sq.shape}")

    # ── numerical comparison ─────────────────────────────────────────────────
    print()
    diff = stats(ref_sq, onnx_sq, label_a=label_ref, label_b=label_test)

    # ── save outputs ─────────────────────────────────────────────────────────
    if args.save_outputs:
        np.save(f"debug_{label_ref.lower().replace('-','_')}_out.npy", ref_sq)
        np.save(f"debug_{label_test.lower()}_out.npy", onnx_sq)
        print("[info] outputs saved as .npy files")

    # ── visualise ────────────────────────────────────────────────────────────
    if args.visualise or args.save_fig:
        visualise(diff, ref_sq, onnx_sq, label_ref, label_test, save_path=args.save_fig)


if __name__ == '__main__':
    main()
