# References and attribution

- [PPO paper](https://arxiv.org/abs/1707.06347), Schulman et al. (2017): clipped policy objective.
- [GAE paper](https://arxiv.org/abs/1506.02438), Schulman et al. (2015): advantage estimator.
- [CleanRL PPO documentation](https://docs.cleanrl.dev/rl-algorithms/ppo/): useful implementation checklist. CritterLab's implementation is independently written; it is not a copied CleanRL module.
- [imitation behavioral cloning documentation](https://imitation.readthedocs.io/en/latest/algorithms/bc.html): supervised imitation of demonstrated actions.
- [MiniGrid](https://minigrid.farama.org/): inspiration for small goal-directed grid worlds. CritterLab uses its own environment and observation implementation; it does not bundle MiniGrid code or assets.
- [PyTorch GRUCell equations](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRUCell.html): reset/update/new gate ordering and recurrent-bias placement for the browser's matching inference.
- [SB3-Contrib recurrent PPO documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html): recurrent-state and episode-start handling reference. The r1 GRU trainer is independently implemented here; no SB3 source is bundled.

All creature, terrain and interface artwork is drawn procedurally for CritterLab.
No external game artwork is bundled. This is AI-assisted original project code,
with established learning algorithms and explicit dependency attribution.
