"""
Track A — Diagnosis harness.

Sweeps the world-model weight W_V and runs ablations to answer:
"Does the world model contribute to navigation, and at what weight?"

Configurations evaluated:
  - W_V sweep (hybrid): map planner + world-model value at increasing weight.
        W_V in {0, 5, 10, 25, 50, 100}   (W_V=0 == map-only baseline)
  - world-model-only (optional, --wm-only): W_DIST=W_BLOCK=W_SUMDIST=0,
        so navigation is driven purely by the learned value of imagined
        latents. Tests whether the world model alone can navigate.

Outputs (to world_model_out/eval/):
  - eval_sweep_<ts>.png   : SR / SPL / avg-steps vs W_V, one line per env
  - eval_summary_<ts>.txt : full numeric table
  - eval_raw_<ts>.json    : per-episode raw results

Usage:
    python eval_batch.py                          # env3, sweep, 10 eps each
    python eval_batch.py --n 20                    # 20 eps per config
    python eval_batch.py --envs 1 2 3              # all three rooms
    python eval_batch.py --wv 0 10 25 50           # custom sweep
    python eval_batch.py --wm-only                 # add world-model-only ablation
    python eval_batch.py --n 20 --envs 1 2 3 --wm-only
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import agent_map_step3 as agent
import MalmoPython

OUT_DIR = "world_model_out/eval"
os.makedirs(OUT_DIR, exist_ok=True)

# Capture the agent's default geometric weights so the hybrid sweep keeps
# them fixed and only varies W_V.
DEF_W_DIST = agent.W_DIST
DEF_W_BLOCK = agent.W_BLOCK
DEF_W_SUMDIST = agent.W_SUMDIST
DEF_V_FADE = agent.V_FADE_DIST


def apply_weights(weights):
    """Set agent module globals so mpc_plan picks them up at call time."""
    for k, v in weights.items():
        setattr(agent, k, v)


def configure_env(env_name):
    agent.ENV = env_name
    agent._cfg = agent._ENV_CONFIGS[env_name]
    agent.MISSION_FILE = agent._cfg["mission"]
    agent.OCCUPIED_CELLS = agent._cfg["occupied"]


def hybrid_config(wv):
    """Map planner + world-model value at weight wv. wv=0 == map-only."""
    return {
        "name": f"WV={wv:g}" if wv > 0 else "map-only (WV=0)",
        "wv": float(wv),
        "is_sweep": True,
        "weights": {
            "W_DIST": DEF_W_DIST,
            "W_BLOCK": DEF_W_BLOCK,
            "W_SUMDIST": DEF_W_SUMDIST,
            "W_V": float(wv),
            "V_FADE_DIST": DEF_V_FADE,
        },
    }


def wm_only_config(wv=50.0):
    """World-model-only: kill geometric terms, navigate by imagined value.
    V_FADE_DIST set huge so V is never faded (no map distance to fade against)."""
    return {
        "name": "WM-only",
        "wv": None,
        "is_sweep": False,
        "weights": {
            "W_DIST": 0.0,
            "W_BLOCK": 0.0,
            "W_SUMDIST": 0.0,
            "W_V": float(wv),
            "V_FADE_DIST": 1e-6,
        },
    }


def run_batch(agent_host, ae, dyn, value, env_name, cfg, n_episodes):
    configure_env(env_name)
    apply_weights(cfg["weights"])

    results = []
    for i in range(n_episodes):
        print(f"\n{'='*60}")
        print(f"  {env_name} | {cfg['name']} | episode {i+1}/{n_episodes}")
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


def compute_spl(results):
    """SPL = (1/N) sum_i  S_i * L_i / max(P_i, L_i)
    L_i = spawn Manhattan distance (optimal lower bound), P_i = path length."""
    if not results:
        return 0.0
    total = 0.0
    for r in results:
        if r["success"]:
            L = max(r["spawn_dist"], 1.0)
            P = max(r["path_length"], L)
            total += L / P
    return total / len(results)


def summarize(results):
    succ = [r for r in results if r["success"]]
    n = max(len(results), 1)
    return {
        "n": len(results),
        "n_success": len(succ),
        "sr": len(succ) / n,
        "avg_steps": float(np.mean([r["steps"] for r in results])) if results else 0.0,
        "std_steps": float(np.std([r["steps"] for r in results])) if results else 0.0,
        "avg_steps_success": float(np.mean([r["steps"] for r in succ])) if succ else 0.0,
        "spl": compute_spl(results),
    }


def make_plots(all_results, envs, sweep_cfgs, wm_cfg, ts):
    """SR / SPL / avg-steps vs W_V, one line per env. WM-only drawn as a star."""
    env_labels = [f"env{e}" for e in envs]
    wv_values = [c["wv"] for c in sweep_cfgs]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Track A — World-Model Contribution vs W_V", fontsize=14, weight="bold")

    metrics = [
        ("sr", "Success Rate", lambda s: s["sr"] * 100, "%"),
        ("spl", "SPL", lambda s: s["spl"], ""),
        ("avg_steps_success", "Avg Steps (success)", lambda s: s["avg_steps_success"], ""),
    ]

    colors = plt.cm.viridis(np.linspace(0, 0.85, len(env_labels)))

    for ax, (mkey, mtitle, mget, munit) in zip(axes, metrics):
        for env, color in zip(env_labels, colors):
            ys = []
            for c in sweep_cfgs:
                s = summarize(all_results[(env, c["name"])])
                ys.append(mget(s))
            ax.plot(wv_values, ys, "o-", color=color, label=env, linewidth=2, markersize=7)

            # WM-only as a star at the right edge (off the W_V axis conceptually)
            if wm_cfg is not None and (env, wm_cfg["name"]) in all_results:
                s = summarize(all_results[(env, wm_cfg["name"])])
                ax.plot(max(wv_values) * 1.12, mget(s), "*", color=color,
                        markersize=16, markeredgecolor="black")

        ax.set_xlabel("W_V (world-model weight)")
        ax.set_ylabel(f"{mtitle} {munit}".strip())
        ax.set_title(mtitle)
        ax.grid(True, alpha=0.3)
        ax.legend(title="(★ = WM-only)" if wm_cfg else None)
        if mkey == "sr":
            ax.set_ylim(0, 105)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"eval_sweep_{ts}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved plot: {out}")
    return out


def make_summary_table(all_results, envs, all_cfgs, ts):
    env_labels = [f"env{e}" for e in envs]
    lines = []
    header = (f"{'Env':<7} {'Config':<18} {'SR':>6} {'AvgSteps':>9} "
              f"{'Std':>7} {'AvgSucc':>9} {'SPL':>7} {'N':>4}")
    lines.append(header)
    lines.append("-" * len(header))
    for env in env_labels:
        for c in all_cfgs:
            res = all_results.get((env, c["name"]), [])
            s = summarize(res)
            lines.append(
                f"{env:<7} {c['name']:<18} {s['sr']:>5.0%} {s['avg_steps']:>9.1f} "
                f"{s['std_steps']:>7.1f} {s['avg_steps_success']:>9.1f} "
                f"{s['spl']:>7.3f} {s['n']:>4}")
        lines.append("")

    table = "\n".join(lines)
    print(f"\n{'='*len(header)}")
    print(table)
    print(f"{'='*len(header)}")

    out = os.path.join(OUT_DIR, f"eval_summary_{ts}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"Saved summary: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="episodes per config")
    parser.add_argument("--envs", type=int, nargs="+", default=[3],
                        help="environments to test (e.g., --envs 1 2 3)")
    parser.add_argument("--wv", type=float, nargs="+",
                        default=[0, 5, 10, 25, 50, 100],
                        help="W_V values to sweep")
    parser.add_argument("--wm-only", action="store_true",
                        help="also run the world-model-only ablation")
    parser.add_argument("--value", default=None,
                        help="override value-head weights (e.g. "
                             "value_head_stepstogo.pth) for A/B comparison")
    args = parser.parse_args()

    if args.value:
        agent.VALUE_WEIGHTS = args.value
        print(f"[A/B] value head override -> {args.value}")

    sweep_cfgs = [hybrid_config(wv) for wv in sorted(args.wv)]
    wm_cfg = wm_only_config() if args.wm_only else None
    all_cfgs = sweep_cfgs + ([wm_cfg] if wm_cfg else [])

    total = len(args.envs) * len(all_cfgs) * args.n
    print(f"Track A diagnosis: {len(args.envs)} env(s) x {len(all_cfgs)} config(s) "
          f"x {args.n} eps = {total} episodes")
    print(f"Sweep W_V: {[c['wv'] for c in sweep_cfgs]}"
          + ("  + WM-only" if wm_cfg else ""))
    print(f"Estimated time: ~{total * 40 / 60:.0f} min\n")

    ae, dyn, value = agent.load_models()
    agent_host = MalmoPython.AgentHost()

    all_results = {}
    ts = time.strftime("%Y%m%d_%H%M%S")

    for env_id in args.envs:
        env_name = f"env{env_id}"
        for cfg in all_cfgs:
            print(f"\n{'#'*60}")
            print(f"  {env_name} | {cfg['name']} | {args.n} episodes")
            print(f"{'#'*60}")
            res = run_batch(agent_host, ae, dyn, value, env_name, cfg, args.n)
            all_results[(env_name, cfg["name"])] = res

            # checkpoint raw results after each config
            dump = {f"{k[0]}|{k[1]}": v for k, v in all_results.items()}
            with open(os.path.join(OUT_DIR, f"eval_raw_{ts}.json"), "w") as f:
                json.dump(dump, f, indent=2, default=str)

    make_summary_table(all_results, args.envs, all_cfgs, ts)
    make_plots(all_results, args.envs, sweep_cfgs, wm_cfg, ts)


if __name__ == "__main__":
    main()
