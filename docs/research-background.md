# Research background: imitation initialization under distribution shift

Literature and repository review, 2026-10-11. This note distinguishes published
results, facts about CritterLab v1, and proposed experiments. It reports no new
experimental findings. The literature search is targeted, not an exhaustive
novelty review.

## What a useful contribution could be

The defensible question is: **When does source-distribution behavioral cloning
help or hurt subsequent PPO adaptation to a target distribution, and which
mechanism explains the difference?** An informative contribution could be a
reproducible boundary between positive and negative transfer in this controlled
navigation family, together with interventions that discriminate explanations.
An adaptive method is a candidate experiment; it is not a contribution merely
because its coefficient changes during training.

BC followed by RL is established. Rajeswaran et al.'s demonstration-augmented
policy gradient method initializes with BC and fine-tunes with an additional
demonstration gradient whose weight decays over iterations. Their manipulation
results motivate this baseline but do not establish the same behavior in
CritterLab. See Section IV-C and its weighting rule in
[Rajeswaran et al., RSS 2018](https://arxiv.org/pdf/1709.10087).

Adaptive suppression of imitation also has prior art. Nair et al. apply BC only
when a learned action-value function estimates that a demonstration action is
better than the actor's action. Their method uses DDPG/HER and continuous
control; a PPO implementation would require explaining the changed estimator
and assumptions. An entropy-gated rule would differ from this Q-filter, but
that alone would not prove novelty or superiority. See Section IV-C and the
ablations in [Nair et al., ICRA 2018](https://arxiv.org/pdf/1709.10089).

## Established mechanisms and their limits

**Imitation distribution shift.** Actions change future inputs. A classifier
trained only on expert trajectories can encounter unfamiliar inputs after its
own errors. DAgger repeatedly collects learner-induced states, obtains expert
labels, and retrains on the aggregate dataset. Its guarantees have assumptions;
they do not promise that a partially observed reactive network can reproduce a
full-state expert. DAgger is an important comparator because it changes data
coverage rather than relying on reward optimization. See Algorithm 3.1 and
Theorems 3.1–3.4 in
[Ross, Gordon and Bagnell, AISTATS 2011](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf).

**PPO and exploration.** The clipped objective is

\[
L^{\mathrm{clip}}(\theta)
=\mathbb{E}_t\left[
\min\left(r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\right)
\right],\qquad
r_t=\frac{\pi_\theta(a_t\mid o_t)}{\pi_{\mathrm{old}}(a_t\mid o_t)}.
\]

Clipping removes incentives for certain excessive probability changes on
sampled actions and permits repeated minibatch updates. The original PPO paper
also includes an entropy bonus. A bonus encourages action randomness; it does
not establish adequate coverage of useful trajectories. See Equations 7 and 9
in [Schulman et al., 2017](https://arxiv.org/pdf/1707.06347).

Clipping is not a hard constraint on the new policy's likelihood ratios or KL
divergence. Shared parameters and optimization across samples can still change
probabilities beyond the nominal interval. Log approximate KL, clip fraction,
gradient norms and value error when diagnosing adaptation failures. The first
claim is analyzed in Section 3 of
[Wang et al., Truly Proximal Policy Optimization](https://arxiv.org/pdf/1903.07940);
the diagnostic recommendation is a proposed design choice here.

**BC and entropy.** In the categorical setting, BC minimizes
\(\mathbb{E}_{(o,a)\sim D}[-\log\pi_\theta(a\mid o)]\). Repeated deterministic
labels can increase confidence on demonstrated inputs. This motivates the
hypothesis that pretraining narrows exploration. It is a hypothesis for this
project, not a finding from the cited PPO paper. Conflicting labels under the
same observation can instead favor a stochastic predictor. Low entropy alone
does not imply poor exploration, and high entropy can increase unsafe moves.

**Reward shaping.** Potential-based shaping has the form
\(F(s,s')=\gamma\Phi(s')-\Phi(s)\), with the relevant terminal-state treatment.
Ng et al. establish policy-invariance conditions for this form. CritterLab v1's
distance difference uses no \(\gamma\) factor, applies to ordinary moves, and
is overwritten by terminal rewards. It should therefore not inherit a
policy-invariance guarantee without a separate argument. Compare explicit
reward conditions and evaluate task success, not just return. See
[Ng, Harada and Russell, ICML 1999](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)
and [the v1 transition implementation](../critterlab/world.py).

## What the current repository does and does not identify

[Protocol v1](protocol.md) fixes a 7 × 7 world, a reactive local observation,
three training seeds, and equal subsequent PPO interaction budgets. BC adds
expert information and optimization. [The results](results.md) preserve the
original seed-held-out estimates and separately disclose a post-hoc
exact-world-overlap audit. This is useful applied evidence; it does not isolate
an exploration mechanism or equal total-resource performance.

The BFS expert uses the whole map, while the actor sees a 3 × 3 neighborhood and
goal displacement. Two hidden situations can share an observation while
requiring different actions. Increasing obstacle density can increase this
aliasing, alter route lengths, and alter the distribution selected by the
solvability rejection filter. A performance drop on denser maps cannot by itself
be assigned to reduced entropy. These are implications of the documented
[world and observation rules](../critterlab/world.py), not measured v1 effects.

The expert also breaks shortest-path ties in north/east/south/west order.
Disagreeing with its single selected label is not always taking a suboptimal
action. A useful diagnostic compares an action with the set of safe shortest
actions; retain deterministic tie handling for reproducibility.

## Experiment distinctions to fix before running

1. **Adaptation:** collect BC demonstrations on source worlds; run PPO on each
   target distribution; evaluate on fresh target worlds. This tests whether
   source initialization helps target learning.
2. **Zero-shot generalization:** train PPO only on source worlds; evaluate on
   shifted worlds without target updates. This tests transfer at evaluation.
3. **Imitation covariate shift:** compare expert-visited and learner-visited
   inputs within a task distribution. This concerns action-induced coverage.

These effects can coexist, but should receive separate names and estimands.
Changing only the evaluation maps does not test PPO's exploration during target
adaptation. Procedural held-out levels are an established approach to RL
generalization evaluation; see Sections 3.2–3.3 of
[Cobbe et al., Procgen, ICML 2020](https://arxiv.org/pdf/1912.01588).
Distinct generator seeds should additionally be checked for duplicate world
content before training; v1 already demonstrates why.

## Recommended controls and accounting

Start with a compact, declared comparison: BC, PPO, BC → PPO, and DAgger. An
adaptive exploration or imitation condition should be added with simple fixed
and time-based alternatives. A full factorial over every demonstration,
architecture and reward option can obscure interpretation; use a primary
comparison and a small number of motivated ablations.

There is no universal exchange rate between an expert query, a simulator step
and a gradient update. Publish a resource vector for each run: demonstration
transitions, expert queries, environment interactions by stage, actor/critic
updates, training wall time and model size. State which dimensions are matched.
Count DAgger learner rollouts and expert labels. Report a PPO-only condition
given a declared additional resource allowance as a sensitivity analysis;
calling it equal compute requires actual compute accounting.

Use the same held-out worlds across methods and paired training-seed blocks.
If each block receives the same demonstrations, inference is conditional on
that demonstration set. Vary independently generated demonstration datasets
across blocks when estimating sensitivity to demonstration sampling. Fix final
checkpoint selection and development-only tuning rules before inspecting the
confirmatory holdout. A written protocol created after a pilot is a protocol
for the next run, not retrospective preregistration of the pilot.

Report the per-seed paired difference in success, not only separate method
means. A declared 10–20 seeds is a useful first target, not a guarantee of
adequate precision. Hundreds of evaluation episodes do not become hundreds of
independently trained agents. Seed-level intervals on a fixed world panel
describe training variation conditional on that panel; make any additional
world-sampling uncertainty explicit.

Agarwal et al. show that few-run point estimates can support misleading RL
comparisons, advocate interval estimates, and explain stratified bootstrap
evaluation across tasks. Their analysis also warns that bootstrap intervals
can under-cover with very few runs. It does not supply a universal sufficient
seed count for this project. See Sections 3–4 and Appendix A of
[Agarwal et al., NeurIPS 2021](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf).

## Proposed mechanism tests, not current conclusions

| Explanation | Diagnostic | Intervention or control | Limit of interpretation |
|---|---|---|---|
| BC narrows exploration | Initial and rollout action entropy; distinct states visited; first target successes | BC with increased fixed entropy; temperature adjustment preserving initial greedy ranking; fixed schedule versus adaptive schedule | Entropy changes action risk and training distribution too; improvement supports an explanation, not exclusive causation |
| Local observations hide necessary information | Repeated observations with conflicting expert action sets; oscillation and timeout traces | Full-observation or memory-capable policy under documented resource accounting | Architecture changes capacity and optimization as well as available information |
| Demonstrations cover the wrong states | Expert versus learner observation coverage; learner-state label errors | DAgger with explicit query budget; source versus target demonstrations | Query access is additional information and must be counted |
| Dense shaping dominates the apparent benefit | Success and progress across reward conditions | Fixed reward ablation, keeping evaluation metric unchanged | New rewards define a changed optimization problem |
| Expert labels have arbitrary ties or noise | Shortest-action-set agreement and demonstration success/length | Controlled label corruption; alternative safe expert actions | Randomly corrupted action labels need not form coherent lower-quality trajectories |

Measure entropy both on a fixed common probe set and on each learner's visited
states. Otherwise differing occupancy can create an apparent entropy difference
without a change on the same inputs. Record shortest-path length, Manhattan
distance, detour excess, obstacle/hazard density and topology family so that
the stated shift is observable. Distinguish recorded action randomness from
actual visitation and from greedy evaluation performance.

The public conclusion should follow the observations: positive transfer,
negative transfer, no clear difference, or unresolved uncertainty are all valid
outcomes. An adaptive method that fails is informative if its failure is
reproducible and narrows the explanation. Nothing in this note establishes that
CritterLab has discovered a new learning phenomenon or method.
