"""
Track C — Localization-noise necessity proof.

The thesis question: does the learned world model actually CONTRIBUTE to
control, or is it decoration on a map+A* planner that rides on ground-truth
position? Offline corr says the value head reads distance from pixels at
corr 0.98, but in the nominal eval the hybrid just sits at the map's ceiling,
so the world model looks optional.

This experiment removes the crutch. We inject Gaussian noise (std in blocks)
into the position the MAP PLANNER believes (agent.POS_NOISE_STD), sweeping the
std up. The geometric terms (dist/blocked/sum_dist) and map-building degrade
with noise; the world-model value term reads PIXELS and is immune. Success/SPL
are scored on the TRUE position.

Two arms, identical episode machinery, ONLY W_V differs:
  - map-only  : W_V = 0           (pure geometry, fully exposed to noise)
  - value-MPC : W_V = default     (geometry + imagined value)

If value-MPC stays above map-only as noise grows, the world model supplies
robustness the map cannot -> it is NECESSARY, not decoration. That gap, as a
function of noise, is the headline figure.

Outputs (world_model_out/eval/):
  - noise_sweep_<ts>.png    : SR / SPL vs noise std, solid=value-MPC dashed=map-only
  - noise_summary_<ts>.txt  : numeric table
  - noise_raw_<ts>.json     : per-episode raw results

Usage:
    py -3.9 world_model/eval_noise.py                       # env3, default noise grid, 15 eps
    py -3.9 world_model/eval_noise.py --n 20 --envs 1 2 3
    py -3.9 world_model/eval_noise.py --noise 0 0.5 1 1.5 2
    py -3.9 world_model/eval_noise.py --value value_head_stepstogo.pth   # A/B the head under noise
"""
import os
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from eval_batch import (agent, configure_env, apply_weights, summarize,
                        DEF_W_DIST, DEF_W_BLOCK, DEF_W_SUMDIST, DEF_V_FADE,
                        OUT_DIR)
import MalmoPython

SEED = 42


def arms(wv_value):
    """Three arms compared at every noise level — identical episode machinery,
    only the scoring weights differ:
      map-only  : pure geometry on the (noisy) believed position.
      value-MPC : geometry + world-model value bonus (value is a small add-on
                  to the dominant distance term, so it tracks map-only).
      wm-only   : geometry OFF — navigate purely by imagined value from pixels.
                  The DECISION never reads position, so it should be immune to
                  localization noise. THIS is the real necessity proof: if it
                  stays flat while map-only collapses, the world model supplies
                  what the map cannot.
    """
    common = {"W_DIST": DEF_W_DIST, "W_BLOCK": DEF_W_BLOCK,
              "W_SUMDIST": DEF_W_SUMDIST, "V_FADE_DIST": DEF_V_FADE}
    return [
        {"name": "map-only", "wv": 0.0, "style": "--",
         "weights": {**common, "W_V": 0.0}},
        {"name": "value-MPC", "wv": float(wv_value), "style": "-",
         "weights": {**common, "W_V": float(wv_value)}},
        {"name": "wm-only", "wv": float(wv_value), "style": ":",
         "weights": {"W_DIST": 0.0, "W_BLOCK": 0.0, "W_SUMDIST": 0.0,
                     "W_V": float(wv_value), "V_FADE_DIST": 1e-6}},
    ]


def run_cell(agent_host, ae, dyn, value, env_name, arm, noise, n):
    configure_env(env_name)
    apply_weights(arm["weights"])
    agent.POS_NOISE_STD = float(noise)

    results = []
    for i in range(n):
        print(f"\n{'='*60}")
        print(f"  {env_name} | {arm['name']} | noise={noise:g} | ep {i+1}/{n}")
        print(f"{'='*60}")
        try:
            r = agent.run_episode(agent_host, ae, dyn, value,
                                  save=False, verbose=False)
            tag = "OK " if r["success"] else "FAIL"
            print(f"  [{tag}] steps={r['steps']}  path={r['path_length']:.1f}  "
                  f"spawn_dist={r['spawn_dist']:.1f}")
            results.append(r)
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append({
                "success": False, "steps": agent.MAX_STEPS,
                "path_length": 0.0, "spawn_dist": 1.0,
                "spawn": (0, 0, 0), "final_pos": (0, 0),
                "dist_to_diamond": 99.0,
            })
        time.sleep(2)
    return results


def make_plots(all_results, envs, noises, arm_specs, ts):
    env_labels = [f"env{e}" for e in envs]
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(env_labels)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    fig.suptitle(f"Track C — Navigation vs localization error "
                 f"[{agent.POS_NOISE_MODEL}]   "
                 f"(solid = value-MPC, dashed = map-only)",
                 fontsize=13, weight="bold")

    metrics = [
        ("sr", "Success Rate (%)", lambda s: s["sr"] * 100),
        ("spl", "SPL", lambda s: s["spl"]),
    ]

    for ax, (mkey, mtitle, mget) in zip(axes, metrics):
        for env, color in zip(env_labels, colors):
            for arm in arm_specs:
                ys = [mget(summarize(all_results[(env, arm["name"], nz)]))
                      for nz in noises]
                ax.plot(noises, ys, arm["style"], color=color, marker="o",
                        linewidth=2, markersize=6,
                        label=f"{env} {arm['name']}")
        ax.set_xlabel(f"localization error std (blocks) [{agent.POS_NOISE_MODEL}]")
        ax.set_ylabel(mtitle)
        ax.set_title(mtitle)
        ax.grid(True, alpha=0.3)
        if mkey == "sr":
            ax.set_ylim(0, 105)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"noise_sweep_{ts}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved plot: {out}")
    return out


def make_summary(all_results, envs, noises, arm_specs, ts):
    env_labels = [f"env{e}" for e in envs]
    lines = []
    header = (f"{'Env':<6} {'Arm':<10} {'Noise':>6} {'SR':>6} {'AvgSteps':>9} "
              f"{'AvgSucc':>9} {'SPL':>7} {'N':>4}")
    lines.append(header)
    lines.append("-" * len(header))
    for env in env_labels:
        for arm in arm_specs:
            for nz in noises:
                s = summarize(all_results.get((env, arm["name"], nz), []))
                lines.append(
                    f"{env:<6} {arm['name']:<10} {nz:>6g} {s['sr']:>5.0%} "
                    f"{s['avg_steps']:>9.1f} {s['avg_steps_success']:>9.1f} "
                    f"{s['spl']:>7.3f} {s['n']:>4}")
            lines.append("")
    table = "\n".join(lines)
    print(f"\n{'='*len(header)}\n{table}\n{'='*len(header)}")
    out = os.path.join(OUT_DIR, f"noise_summary_{ts}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"Saved summary: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="episodes per cell")
    ap.add_argument("--envs", type=int, nargs="+", default=[3])
    ap.add_argument("--noise", type=float, nargs="+", default=None,
                    help="localization-error std values (blocks) to sweep; "
                         "default grid depends on --noise-model")
    ap.add_argument("--noise-model", default="drift",
                    choices=["drift", "bias", "jitter"],
                    help="error model: drift=odometry random walk (default), "
                         "bias=constant per-episode offset, jitter=per-step "
                         "white noise (pathological, both arms collapse)")
    ap.add_argument("--wv", type=float, default=agent.W_V,
                    help="W_V for the value-MPC arm (default = agent default)")
    ap.add_argument("--value", default=None,
                    help="override value-head weights (A/B under noise)")
    args = ap.parse_args()

    if args.value:
        agent.VALUE_WEIGHTS = args.value
        print(f"[A/B] value head override -> {args.value}")

    np.random.seed(SEED)
    agent.POS_NOISE_MODEL = args.noise_model
    if args.noise is None:
        args.noise = {"drift":  [0.0, 0.02, 0.05, 0.1, 0.2],
                      "bias":   [0.0, 0.25, 0.5, 1.0, 1.5],
                      "jitter": [0.0, 0.05, 0.1, 0.2, 0.4]}[args.noise_model]
    noises = sorted(args.noise)
    arm_specs = arms(args.wv)

    total = len(args.envs) * len(arm_specs) * len(noises) * args.n
    print(f"Track C: {len(args.envs)} env(s) x {len(arm_specs)} arms x "
          f"{len(noises)} noise x {args.n} eps = {total} episodes")
    print(f"Noise grid: {noises}   value-MPC W_V={args.wv:g}")
    print(f"Estimated time: ~{total * 40 / 60:.0f} min\n")

    ae, dyn, value = agent.load_models()
    agent_host = MalmoPython.AgentHost()

    all_results = {}
    ts = time.strftime("%Y%m%d_%H%M%S")
    for env_id in args.envs:
        env_name = f"env{env_id}"
        for arm in arm_specs:
            for nz in noises:
                print(f"\n{'#'*60}")
                print(f"  {env_name} | {arm['name']} | noise={nz:g} | {args.n} eps")
                print(f"{'#'*60}")
                res = run_cell(agent_host, ae, dyn, value,
                               env_name, arm, nz, args.n)
                all_results[(env_name, arm["name"], nz)] = res

                dump = {f"{k[0]}|{k[1]}|{k[2]}": v
                        for k, v in all_results.items()}
                with open(os.path.join(OUT_DIR, f"noise_raw_{ts}.json"), "w") as f:
                    json.dump(dump, f, indent=2, default=str)

    make_summary(all_results, args.envs, noises, arm_specs, ts)
    make_plots(all_results, args.envs, noises, arm_specs, ts)


if __name__ == "__main__":
    main()
