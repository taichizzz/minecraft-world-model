# Model manifest — Malmo world-model agent

Weights live in `D:\malmo\` (trainers run from there and save relative paths).

**A usable agent = a COHERENT STACK: AE + dynamics + value head all trained on
the SAME latent.** A dynamics model or value head is only valid with the encoder
it was trained on — never mix across AE versions (e.g. `value_head_dist.pth`
only works with `ae_multienv.pth`, NOT with `ae_predictive.pth`).

---

## ⭐ multienv — the cross-environment model ("the 5/28 model")

Trained on all three env datasets (dataset_1/2/3_human) → works in env1/2/3.

| role | file | date |
|------|------|------|
| AE (encoder+decoder) | `ae_multienv.pth` | 2026-05-27 (warmstart aeturn3) |
| dynamics | `dynamics_multienv.pth` | **2026-05-28** |
| value head (optimized) | `value_head_dist.pth` | 2026-05-30 (distance-to-goal) |
| value head (original, weak) | `value_head_multienv.pth` | 2026-05-27 ("diamond visible") |

"Worked for all environments but not optimized" on 5/28 = the stack used
`value_head_multienv` (visibility). The optimization was training
`value_head_dist` (dense distance-to-goal) on 5/30.

## P2b / predictive — sharper imagination, equal control at H=4

Joint encoder+dynamics rebuild for a *predictable* latent.

| role | file | date |
|------|------|------|
| AE | `ae_predictive.pth` | 2026-05-30 |
| dynamics | `dynamics_predictive.pth` | 2026-05-30 |
| value head | `value_head_dist_pred.pth` | 2026-05-30 (distance head on P2b latent) |

Imagines frames ~25–30% sharper than multienv; but imagined-VALUE reliability at
the H=4 planning horizon is identical (0.901). Treat it as "a better world
model," not a control upgrade.

---

## CURRENT AGENT DEFAULT  (`agent_map_step3.py` L145–147)
```
AE   = ae_multienv.pth
DYN  = dynamics_multienv.pth
VAL  = value_head_dist.pth
```
= the multienv stack with the optimized value head. **This is the 5/28
all-environment model, improved.** This is your best general model right now.

## Other recent value heads (all on the ae_multienv latent)
- `value_head_stepstogo.pth` (05-30) — steps-to-go / return-to-go variant.
- `value_head_smooth.pth` (05-26) — on the OLD aeturn3 latent (pre-multienv); not compatible with ae_multienv.

## Experiment — NOT for deployment
- `dynamics_multienv_v2.pth` (05-30) — P2a "harder dynamics" test; measured WORSE than `dynamics_multienv`. Discard.

## Legacy (pre-2026-05-27) — single-env iterations, superseded
`ae.pth`/`encoder.pth`, `ae2–6`/`encoder2–6`, `dynamics(_multistep|_balanced|2)*`,
`aeturn1–3`/`encoderturn1–3`, `dynamics_turn_*`, `dynamics_multistep_k*`,
`value_head_1/2`, `dynamics_turn_human*`. Kept for history; the current agent
does not use any of these.
