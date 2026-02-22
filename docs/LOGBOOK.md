# Logbook

## 2026-02-22
- Added an edge-adapter conditioning mode in `UNet` with multi-scale condition injection at skip and middle features.
- Updated Flow Matching model dispatch so `edge_adapter` mode passes condition separately instead of channel concatenation.
- Simplified CelebA edge conditioning to Canny/XDoG-focused mixed edges with per-method weighting and method-level sigma overrides.
- Added new HW4 edge-conditioning configs (including adapter and tuned variants) for Modal dry-run and 2-GPU runs.
- Refreshed edge-comparison experiment scripts, added extended visualization utilities, and removed older PiDiNet-specific experiment scripts.
- Added new edge detector package stubs/assets and generated output grids for edge-method comparisons and sample previews.
