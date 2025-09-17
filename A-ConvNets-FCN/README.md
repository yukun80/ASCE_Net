# A-ConvNets-FCN Comparison Pipeline

This module implements the A-ConvNets + FCN aggregation network described in the reference paper, using the existing five-channel ASC labels as source data. The workflow converts 5-channel labels into sparse 2-channel maps (amplitude `A`, phase `alpha`), trains the comparison model, and provides inference/evaluation utilities.

## Directory Layout

```
A-ConvNets-FCN/
  config.yaml              # Paths and hyper-parameters
  convert_labels.py        # Offline 5ch -> 2ch conversion
  dataset.py               # Dataset and helper conversions
  model.py                 # A-ConvNets backbone with FCN heads
  losses.py                # Full + non-zero MSE loss
  train.py                 # Training loop with validation split
  infer.py                 # Single-image / batch inference
  eval.py                  # Dataset-level evaluation metrics
  utils/
    io.py                  # Config loading and path helpers
    seed.py                # Reproducibility helpers
    peaks.py               # Peak detection utilities
  outputs/                 # Checkpoints, logs (populated at runtime)
```

## Usage

1. **Convert labels** (once per dataset):
   ```bash
   python convert_labels.py --config config.yaml
   ```

2. **Train the comparison model**:
   ```bash
   python train.py --config config.yaml
   ```
   Use `--label-on-the-fly` if you prefer converting labels during loading instead of offline conversion.

3. **Run inference** (single file or an entire folder). Omit `--raw` to use `data.sar_root` from the config:
   ```bash
   python infer.py --config config.yaml --checkpoint outputs/checkpoints/aconv_fcn_best.pt [--raw <file-or-dir>]
   ```

4. **Evaluate on the dataset**:
   ```bash
   python eval.py --config config.yaml --checkpoint outputs/checkpoints/aconv_fcn_best.pt
   ```

Adjust paths/hyper-parameters through `config.yaml` as needed. All relative paths are resolved against the project root inferred from the config file.