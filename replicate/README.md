# channel_entropy replication

Trains the channel_entropy model from scratch on CIFAR-10 using this repo
(`conv` branch). Runs 5 seeds and plots test accuracy over training epochs.

## Architecture

**channel_entropy** = `ChannelWBack` + `Conv2DRecurrentDiscrete(entropy_beta, lambda_entropy=1)`

| Layer | Module | Description |
|-------|--------|-------------|
| Win | `Conv2D` | 3→16 ch, 5×5 conv, feedforward |
| J1 | `Conv2DRecurrentDiscrete` | 16 ch recurrent, 5×5, entropy modulation built in |
| W_back | `ChannelWBack` | FC(10→16) broadcast over 32×32, frozen |
| W_out | `PooledFlattenFC` | 8×8 avg-pool + flatten + FC(256→10) |

`ChannelWBack` and `PooledFlattenFC` are defined in `darnax.modules.conv.spatial_fc`.

## Setup

```bash
conda activate darnn          # or whatever env has jax, equinox, optax, torch

# optional: install as editable package
pip install -e /path/to/darnax/
```

No separate installation is needed — the script adds `../src` to `sys.path` at
runtime so the package modules are picked up directly.

## Run

```bash
cd replicate/
python replicate_channel_entropy.py
```

Trains 5 seeds × 20 epochs. After each epoch: evaluates model head accuracy and
runs a linear probe (20 epochs) on pooled J1 representations.
Saves `accuracy_curves.png` (two panels: head accuracy and probe accuracy over epochs).

Expected runtime: ~20–40 min on a single GPU.

## Hyperparameters

Loaded from `best_channel_entropy_cfg.json` (same directory):

| Key | Value | Description |
|-----|-------|-------------|
| `lr_j` | 9.74e-4 | J1 learning rate |
| `lr_win` | 0.0226 | Win learning rate |
| `lr_wout` | 0.0432 | W_out learning rate |
| `kernel_decay_rate` | 7.85e-4 | Per-epoch kernel L2 decay |
| `threshold_j` | 1.98 | J1 Hebbian threshold κ |
| `threshold_win` | 0.85 | Win Hebbian threshold |
| `j_d` | 0.895 | J1 self-coupling diagonal |
| `entropy_beta` | 0.345 | Entropy modulation temperature |
| `momentum` | 0.316 | SGD momentum |
| `clamped_n_iter` | 5 | Clamped phase iterations |
| `free_n_iter` | 6 | Free phase iterations |
| `strength_back` | 1.47 | W_back output scale |

## Training dynamics

Each batch: `warmup(1) → clamped(5) → free(6) → Hebb update`.
After each batch: Win filters normalised to unit L2 norm per output channel.
After each epoch: Win and J1 kernels multiplied by `(1 − kernel_decay_rate)`.
W_back is frozen throughout.

## Notes on accuracy

Two accuracy metrics are reported per epoch:

- **Model head** (`W_out`): test accuracy from the model's own `PooledFlattenFC`
  classification head.
- **Linear probe**: `Linear(256, 10, bias=False)` trained for 20 epochs (Adam,
  lr=1e-3, wd=1.433e-4) on 8×8 avg-pooled J1 representations (→ 256-dim).
  Reference best probe accuracy: 0.4501 (shown as dashed line in the plot).
