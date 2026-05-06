# channel_entropy replication

Trains the best channel_entropy model from T21 from scratch on CIFAR-10 using
`darnax_update` (this repo, `conv` branch). Runs 5 seeds and plots test accuracy
over training epochs.

## Architecture

**channel_entropy** = `ChannelWBack` + `Conv2DRecurrentDiscrete(entropy_beta, lambda_entropy=1)`

| Layer | Module | Description |
|-------|--------|-------------|
| Win | `Conv2D` | 3→16 ch, 5×5 conv, feedforward |
| J1 | `Conv2DRecurrentDiscrete` | 16 ch recurrent, 5×5, entropy modulation built in |
| W_back | `ChannelWBack` | FC(10→16) broadcast over 32×32, frozen |
| W_out | `PooledFlattenFC` | 8×8 avg-pool + flatten + FC(256→10) |

The original sweep defined an `EntropyJ1` subclass to add entropy modulation.
This is not needed here — `Conv2DRecurrentDiscrete` in `darnax_update` accepts
`entropy_beta` and `lambda_entropy` directly.

**Two custom subclasses remain** (non-conv, in the script itself):
- `ChannelWBack` — reshape logic for label→J1 feedback
- `PooledFlattenFC` — spatial pooling before the FC readout

## Setup

```bash
# from darnn_hpc/ root
conda activate darnn          # or whatever env has jax, equinox, optax, torch

# optional: install darnax_update as editable package
pip install -e darnax_update/
```

No separate installation is needed if you already have the original `darnax`
package active — the script adds `darnax_update/src` to `sys.path` at runtime
so the updated conv modules take precedence.

## Run

```bash
cd darnax_update/replicate
python replicate_channel_entropy.py
```

Trains 5 seeds × 20 epochs, prints per-epoch model-head accuracy, then runs a
linear probe (20 epochs) on the final J1 representations per seed.
Saves `accuracy_curves.png` (two panels: head accuracy over epochs + probe bar chart).

Expected runtime: ~20–40 min on a single GPU (same as the original sweep).

## Hyperparameters

Loaded from `darnn_hpc/logs/91_win_wback_sweep/best_channel_entropy_cfg.json`:

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

Two accuracy metrics are reported:

- **Model head** (`W_out`): per-epoch test accuracy from the model's own `PooledFlattenFC`
  classification head. Tracked every epoch.
- **Linear probe**: after all epochs, a `Linear(256, 10, bias=False)` is trained for
  `PROBE_EPOCHS=20` epochs (Adam, lr=1e-3, `probe_wd=1.433e-4`) on pooled J1
  representations (8×8 avg-pool → 256-dim). This matches the T21 evaluation protocol.
  The T21 best result is `probe_acc=0.4501`; the replicate script plots this as a
  reference line in the probe bar chart.
