# 4.4 Execution and Approaches

The following are the approaches and solutions we have tried and tested. All the observations are explored in Section 4.5.

## Residual Dynamics

As of currently, our model is learning from

  z_{t+1} = f(z_t, a_t)

Since the movement in our controlled environment is very small, we can actually try to predict the change instead of predicting the whole new latent state:

  z_{t+1} = z_t + f(z_t, a_t)

This might be able to reduce the compounding errors that are occurring and might stabilize the rollout.

## Multi-Step Prediction

To improve prediction stability, a multi-step prediction approach is tested. Instead of predicting only the next state, the model recursively predicts multiple future states during training (unrolling k steps and applying the loss at every step). This approach can:

1. Reduce error accumulation
2. Improve temporal consistency
3. Enhance long-term prediction performance

## Adding Temporal Smoothness

The current world model predicts short-term dynamics reasonably well and preserves scene structure over multiple steps, but fine visual details fade over rollout. We measured latent smoothness and found that reconstruction-only autoencoders can produce temporally unstable latent spaces, especially when adding more datasets. After adding a temporal smoothness term to the autoencoder loss, rollout stability improved:

  L = L_recon + beta * || z_{t+1} - z_t ||^2

An important lesson emerged later from this line of work: temporal smoothness makes the latent space *stable* (consecutive latents stay close), but stability is not the same as *action-conditioned predictability* (whether z_{t+1} is a learnable function of (z_t, a_t)). Diagnosing and fixing that distinction became the basis of the predictable-latent rebuild described in Section 4.7.

## Value Head and Diamond Visibility

In addition to the autoencoder and dynamics model, a value head estimates the usefulness of a latent state for reaching the goal, so that imagined future latent states can be scored during planning. The first version of this head predicted whether the diamond is visible in the current frame, supervised by a color-based (HSV) diamond detector. The detector thresholds hue and saturation only — hue is mathematically invariant to brightness — so detection survives lighting changes between day and night frames in our dataset. The value head itself went through several major revisions as our understanding improved; the full evolution and its evaluation are presented in Section 4.9.

## Closed-Loop Navigation: From A* Baseline to MPC

After offline rollout testing, the project moved into live closed-loop navigation. This stage is more difficult because the agent must not only predict future states, but also choose useful actions in real time.

First, an occupancy map and an A* planner were implemented as a classical baseline. The purpose of A* was not to replace the world model, but to debug the environment, action system, map building, and diamond localization. If the A* baseline cannot solve the environment, the issue is likely in map construction, goal detection, or action conversion rather than in the learned dynamics model.

We then replaced the A* controller with a model-predictive control (MPC) approach over the learned dynamics model. The agent enumerates all possible action sequences of length H. With three actions and horizon H = 4, the controller evaluates 3^4 = 81 candidate action sequences at each timestep. For each sequence, the system performs two types of prediction in parallel:

1. A kinematic simulation of the agent's future position and yaw on the occupancy map (exact geometry, no learning), and
2. A latent rollout using the learned dynamics model, where the value head scores every imagined step.

The score of a candidate sequence combines geometric terms with the discounted imagined value:

  Score = -100 * dist  -  10 * blocked  -  1 * sum_dist  +  w_v * SUM_{h=0..H-1} gamma^h * V(z_hat_h)

where gamma = 0.95, w_v fades linearly to zero within 3 blocks of the target (the map handles the final approach), and z_hat_h is the imagined latent at step h. The agent executes only the first action of the best sequence and replans at the next frame (receding horizon).

Two refinements to the distance term were introduced during closed-loop testing. First, `dist` was originally the straight-line (Manhattan) distance to the target, which measures distance *through* obstacles; it was replaced by the geodesic (BFS) distance on the believed occupancy map, which measures the distance *around* known obstacles. Second, the way the agent acquires and remembers the goal location was substantially reworked after instrumented failure analysis; this is covered together with the closed-loop evaluation campaign (see the closed-loop evaluation section).

## Human Demonstration Motivation

An important observation from the closed-loop experiments was that random exploration data alone was not sufficient for useful goal-directed planning. In some runs, the system was able to visually detect the diamond block, but the resulting navigation behavior still failed to move toward it consistently. The agent could continue turning or moving in less useful directions. This suggested that while the model had learned some environment dynamics, it lacked enough successful, target-oriented trajectories to support reliable planning.

To address this, the data collection process was extended to include human-controlled demonstrations, in which the agent turns appropriately, aligns with the corridor or target, and moves directly toward the diamond.

This decision paid off. The final demonstration corpus contains 600 human episodes across the three environments (200 per environment, 20,611 frames in total), of which 543 episodes (90%) successfully reach the goal. Every model version described from Section 4.5 onward — the multi-environment autoencoder, the predictable-latent rebuild, the value-consistent dynamics, and all value heads — is trained on this corpus. The ground-truth positions and headings stored with each episode also enabled the exact cost-to-go labels used by the final value head (Section 4.9).


# 4.5 Training and Observations

## Testing the first Visual Encoder

(Reference to 4.1, Figure 4.1.) Since the reconstruction is just a plain gradient, we can conclude that the dataset is not good enough for the visual encoder. Therefore, we reset the sample mission to recollect the data.

## Fixing the AE

Based on the limitations observed in the initial dataset, we redesigned the mission to improve visual diversity and encourage more informative observations, limiting the Minecraft environment with distinctive walls so that different viewpoints do not look identical. As shown in Figure 4.2, we achieved a reasonable result after retraining: the reconstruction quality indicates that the latent space captures spatial layout, object identity (diamond block, walls), and relative geometry. Minor blurring suggests compression but no catastrophic loss of structure.

## Residual Dynamics Test

(Figure 4.4.) Despite the expectation, the result came out worse than our initial result using baseline dynamics. We conclude that residual dynamics performed worse under our current setup.

## Multi-Step Dynamics Test

(Figures 4.5–4.7.) Since residual dynamics did not work, we tried to improve the existing model by training with a multi-step loss. The first attempts were unstable — rollouts hallucinated and blew up — which we addressed with gradient clipping, a reduced learning rate, and delta scaling. After these adjustments we eliminated catastrophic divergence:

| Model | Latent MSE | Pixel MSE |
|---|---|---|
| Baseline 1-step | 0.160147 | 0.000822 |
| Residual | 10,982,221 | 0.1082 |
| Multi-step (after adjustments) | ~6.66 | 0.0492 |

## Testing After Adding Temporal Smoothness

(Tables as previously reported for ds1/ds2, baseline vs. multistep.) The multistep dynamics model outperformed the baseline on both datasets: although its 1-step prediction on dataset 1 became slightly worse, it achieved noticeably lower rollout latent and pixel errors, showing much better long-horizon stability. On dataset 2 the multistep model improved both 1-step and rollout performance. Overall, multistep training is more effective than baseline 1-step training for maintaining accurate predictions over time.

## Dataset Expansion Observation

To evaluate what the model could do so far, we designed a new environment (Dataset 3) and repeated the process. After the success and stability of dataset 3, we designed more environments and attempted the same. However, when datasets 4 and 5 were added, performance on the previously learned environments began to degrade: rollout pixel error on datasets 1–3 generally became worse than the best model trained up to dataset 3.

## Resolution: Multi-Environment Joint Training

The degradation problem was ultimately resolved not by adding ever more datasets, but by changing how we train and evaluate:

1. **Joint multi-environment training.** Instead of sequentially adding datasets, we trained a single autoencoder (warm-started from the best dataset-3 model) jointly on the human-demonstration corpora of all three environments, with the temporal-smoothness term and an additional temporal-contrastive (InfoNCE) objective. The corresponding dynamics model was then trained on this shared latent space. This produced one world model that serves all three environments simultaneously, instead of one model that degrades as environments are appended. (Figure 4.9 — training curves.)

2. **A stricter evaluation standard.** Aggregate rollout MSE can hide failure modes, so we adopted a per-action gate: for each action type (move, turn-right, turn-left), the decoded next-frame error of the dynamics prediction must beat a *persistence* baseline that simply predicts "nothing changed." A dynamics model that fails this test cannot imagine the consequence of that action at all. The multi-environment model passes it for every action (decoded next-frame MSE ≈ 0.008–0.009 vs. persistence 0.011–0.015), although the gap to the encoder ceiling (≈ 0.0013) showed substantial remaining headroom — which motivated the latent-space diagnosis in Section 4.7. (Figure 4.10 — per-action comparison.)

3. **Generalization measured as a capability, not assumed from training data.** Rather than asking the model to absorb every new environment into training, we later evaluated the finished system zero-shot on newly designed room layouts it had never trained on (using the same block vocabulary). The closed-loop evaluation section reports 90–100% success on these unseen layouts — evidence that the combination of joint training and the improvements in Sections 4.7–4.9 generalizes across environment *forms* without retraining.

This reframing — one jointly trained world model, gated per action against a persistence baseline, and tested zero-shot on unseen layouts — is the foundation every subsequent section builds on.
