# Model manifest — Malmo world-model agent (updated 2026-06-11)

Weights live in `D:\malmo\`. **A usable agent = a COHERENT STACK: AE + dynamics +
value head trained on the SAME latent.** Never mix heads/dynamics across AE
versions. All current stacks share the `ae_predictive.pth` latent.

---

## ⭐ CURRENT BEST — deployable hybrid agent ("p2b_vc", WV50)

| role | file |
|------|------|
| AE | `ae_predictive.pth` |
| dynamics | `dynamics_predictive_vc.pth` (value-consistent: imagined-value corr@h4 0.963) |
| value head | `value_head_dist_pred.pth` (distance; corr 0.98 from pixels) |

Run via: `eval_grid.py --stacks p2b_vc --wv 50`.
Closed-loop (new goal-memory planner, paired spawns): **env3 100%**,
unseen layouts **env4 90% / env5 100% / env6 100%** (SPL .358/.587/.581).
env4 residual failures = 30 s mission clock, not logic.

### Variant: appearance-robust head ("p2b_aug")
Swap value head → `value_head_dist_pred_aug.pth`: goal-reading survives
±40% brightness/±30% contrast/±20% color (corr 0.92 vs 0.78 clean-trained),
costs ~nothing clean (0.969 vs 0.978). Use when lighting varies.

### Variant: value-only champion (WM drives alone — capability demo)
`ae_predictive` + `dynamics_predictive_vc_stg.pth` + `value_head_stg_aug.pth`
(steps-to-go cost field, γ=0.9; obstacle+turn aware: residual corr .55/.49/.68
vs dist head's .11/.18/.23). Value-only scoring: **33% SR** in env3, successes
faster than the map planner (33 vs 44 steps). This is the THESIS capability
number, not the deployment config.

---

## The planner (system side, agent_map_step3.py — as important as the weights)
2026-06-10 goal-memory fixes (mark placement sees over obstacles + blob-area
range estimate; evidence-based forgetting; glimpse hints; 360° sweeps) +
BFS-geodesic distance + step_kinds/goal_events logging + paired-spawn support.
These took unseen-layout SR from 50/85/95 → 90/100/100.

⚠ The module DEFAULTS (AE_WEIGHTS etc. in agent_map_step3.py) still point to
the LEGACY multienv stack — eval scripts override per run. Flip defaults to
p2b_vc if running the agent directly.

---

## Negative results — keep for the report, do NOT deploy
- `value_head_stg_lin.pth` + `dynamics_predictive_vc_lin.pth` — linear-label
  cost field: passed every offline gate, **7% closed-loop** (near-goal gradient
  mattered; far field is partial-observability-limited). Lesson: offline corr
  does not predict closed-loop driving.
- `ae_predictive_aug.pth` + `dynamics_predictive_aug.pth` — joint encoder+aug
  training DIVERGED (unbounded latent). Garbage weights; safe pattern is
  frozen-encoder + augmented head.
- `dynamics_multienv_v2.pth` — P2a hardened training; worse than baseline.

## Legacy (superseded)
- multienv stack: `ae_multienv` + `dynamics_multienv` + `value_head_dist`
  ("the 5/28 model") — first cross-env model; superseded by p2b_vc.
- `value_head_multienv` (visibility; 0% as navigator), `value_head_stepstogo`
  (old latent), everything pre-2026-05-27 (ae1-6, aeturn*, dynamics_turn_*,
  dynamics_multistep_k*, value_head_1/2...).
