"""
Best-combination sweep — model stack x scoring weights, closed-loop SR/SPL.

Answers "which models + which MPC scoring weights navigate best?". Only
COHERENT stacks are allowed (the value head + dynamics must match the encoder
that produced the latent):
  old : ae_multienv   + dynamics_multienv   + value_head_dist
  p2b : ae_predictive + dynamics_predictive + value_head_dist_pred

Grid = stacks x W_V [x GAMMA_V]. Geometric weights stay at the agent defaults
(already tuned). map-only (W_V=0) is run ONCE — it ignores the world model, so
it is stack-independent. Configs are ranked by SPL (tie-break SR).

Optionally evaluate the combos UNDER degraded localization with --noise /
--noise-model — that is the thesis-relevant regime (does the world-model value
term earn its keep when position is no longer trustworthy?).

Outputs (world_model_out/eval/):
  combo_rank_<ts>.txt   ranked table
  combo_<ts>.png        SR + SPL bar chart (ranked)
  combo_raw_<ts>.json   per-episode raw

NOTE: closed-loop — needs the Malmo client + GPU. Launch yourself.

Usage:
  py -3.9 world_model/eval_grid.py --n 20 --envs 3
  py -3.9 world_model/eval_grid.py --stacks old p2b --wv 0 50 100 --n 15
  py -3.9 world_model/eval_grid.py --noise 0.1 --noise-model drift --n 15
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

STACKS = {
    "old": ("ae_multienv.pth", "dynamics_multienv.pth", "value_head_dist.pth"),
    "p2b": ("ae_predictive.pth", "dynamics_predictive.pth", "value_head_dist_pred.pth"),
}


def load_stack(name):
    ae_w, dyn_w, val_w = STACKS[name]
    agent.AE_WEIGHTS, agent.DYN_WEIGHTS, agent.VALUE_WEIGHTS = ae_w, dyn_w, val_w
    return agent.load_models()


def weights_for(wv, gamma):
    return {"W_DIST": DEF_W_DIST, "W_BLOCK": DEF_W_BLOCK,
            "W_SUMDIST": DEF_W_SUMDIST, "V_FADE_DIST": DEF_V_FADE,
            "W_V": float(wv), "GAMMA_V": float(gamma)}


def build_configs(stacks, wvs, gammas):
    """map-only once (stack-independent); value-MPC over stacks x wv>0 x gamma."""
    configs = []
    if any(w <= 0 for w in wvs):
        configs.append({"stack": stacks[0], "label": "map-only",
                        "weights": weights_for(0.0, gammas[0])})
    for stack in stacks:
        for wv in wvs:
            if wv <= 0:
                continue
            for gamma in gammas:
                gtag = f" g{gamma:g}" if len(gammas) > 1 else ""
                configs.append({"stack": stack,
                                "label": f"{stack} WV{wv:g}{gtag}",
                                "weights": weights_for(wv, gamma)})
    return configs


def run_cell(agent_host, ae, dyn, value, env_name, weights, n, label):
    configure_env(env_name)
    apply_weights(weights)
    results = []
    for i in range(n):
        print(f"\n{'='*60}\n  {env_name} | {label} | ep {i+1}/{n}\n{'='*60}")
        try:
            r = agent.run_episode(agent_host, ae, dyn, value,
                                  save=False, verbose=False)
            tag = "OK " if r["success"] else "FAIL"
            print(f"  [{tag}] steps={r['steps']}  path={r['path_length']:.1f}  "
                  f"spawn_dist={r['spawn_dist']:.1f}")
            results.append(r)
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append({"success": False, "steps": agent.MAX_STEPS,
                            "path_length": 0.0, "spawn_dist": 1.0,
                            "spawn": (0, 0, 0), "final_pos": (0, 0),
                            "dist_to_diamond": 99.0})
        time.sleep(2)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="episodes per config")
    ap.add_argument("--envs", type=int, nargs="+", default=[3])
    ap.add_argument("--stacks", nargs="+", default=["old", "p2b"],
                    choices=list(STACKS))
    ap.add_argument("--wv", type=float, nargs="+", default=[0.0, 50.0],
                    help="W_V values to sweep (0 = map-only baseline)")
    ap.add_argument("--gamma", type=float, nargs="+", default=[agent.GAMMA_V],
                    help="GAMMA_V values to sweep")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="localization-error std (0 = perfect localization)")
    ap.add_argument("--noise-model", default="drift",
                    choices=["drift", "bias", "jitter"])
    args = ap.parse_args()

    np.random.seed(SEED)
    agent.POS_NOISE_MODEL = args.noise_model
    agent.POS_NOISE_STD = float(args.noise)

    configs = build_configs(args.stacks, args.wv, args.gamma)
    total = len(args.envs) * len(configs) * args.n
    noise_tag = (f"  [noise={args.noise:g} {args.noise_model}]"
                 if args.noise > 0 else "")
    print(f"Combo sweep: {len(args.envs)} env x {len(configs)} configs x "
          f"{args.n} eps = {total} episodes{noise_tag}")
    print(f"  stacks={args.stacks}  wv={args.wv}  gamma={args.gamma}")
    print(f"  configs: {[c['label'] for c in configs]}")
    print(f"  est ~{total * 40 / 60:.0f} min\n")

    agent_host = MalmoPython.AgentHost()
    ts = time.strftime("%Y%m%d_%H%M%S")
    all_results = {}

    for env_id in args.envs:
        env_name = f"env{env_id}"
        loaded = {}
        for cfg in configs:
            if cfg["stack"] not in loaded:
                loaded[cfg["stack"]] = load_stack(cfg["stack"])
            ae, dyn, value = loaded[cfg["stack"]]
            print(f"\n{'#'*60}\n  {env_name} | {cfg['label']} | {args.n} eps\n{'#'*60}")
            res = run_cell(agent_host, ae, dyn, value, env_name,
                           cfg["weights"], args.n, cfg["label"])
            all_results[(env_name, cfg["label"])] = res
            with open(os.path.join(OUT_DIR, f"combo_raw_{ts}.json"), "w") as f:
                json.dump({f"{k[0]}|{k[1]}": v for k, v in all_results.items()},
                          f, indent=2, default=str)

    rows = [(env_name, label, summarize(res))
            for (env_name, label), res in all_results.items()]
    rows.sort(key=lambda r: (r[2]["spl"], r[2]["sr"]), reverse=True)

    header = f"{'Env':<6} {'Config':<18} {'SR':>6} {'SPL':>7} {'AvgSucc':>9} {'N':>4}"
    lines = [f"RANKED BY SPL{noise_tag}", header, "-" * len(header)]
    for env_name, label, s in rows:
        lines.append(f"{env_name:<6} {label:<18} {s['sr']:>5.0%} {s['spl']:>7.3f} "
                     f"{s['avg_steps_success']:>9.1f} {s['n']:>4}")
    table = "\n".join(lines)
    print(f"\n{'='*len(header)}\n{table}\n{'='*len(header)}")
    with open(os.path.join(OUT_DIR, f"combo_rank_{ts}.txt"), "w",
              encoding="utf-8") as f:
        f.write(table + "\n")
    best = rows[0]
    print(f"\n  BEST: {best[0]} / {best[1]}  ->  "
          f"SPL {best[2]['spl']:.3f}  SR {best[2]['sr']:.0%}")

    labels = [f"{e}\n{l}" for e, l, _ in rows]
    srs = [s["sr"] * 100 for _, _, s in rows]
    spls = [s["spl"] for _, _, s in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(max(10, 1.3 * len(rows)), 5.2))
    fig.suptitle(f"Best-combination sweep{noise_tag}  (ranked by SPL)",
                 fontsize=13, weight="bold")
    axes[0].bar(x, srs, color="tab:blue"); axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_title("Success Rate"); axes[0].set_ylim(0, 105)
    axes[1].bar(x, spls, color="tab:green"); axes[1].set_ylabel("SPL")
    axes[1].set_title("SPL")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"combo_{ts}.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
