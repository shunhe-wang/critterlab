"""Recurrent training contracts, independent analytic GAE, and sequence layout."""
import json
from pathlib import Path
import torch
from critterlab.model import Policy
from critterlab.navigation_model import GardenGRU
from critterlab.navigation_train import gae, chunks, demonstrations, clone, ppo, prefix_state, cycle

def test_gae_terminal_boundary_matches_hand_calculation():
    reward = torch.tensor([[1.],[2.]])
    value = torch.tensor([[.5],[.25]])
    result = gae(reward,value,torch.tensor([[0.],[1.]]),torch.tensor([99.]))
    torch.testing.assert_close(result,torch.tensor([[2.393375],[1.75]]))
    result = gae(reward,value,torch.ones(2,1),torch.tensor([99.]))
    torch.testing.assert_close(result,torch.tensor([[.5],[1.75]]))

def test_chunks_preserve_environment_and_time_order():
    source = torch.tensor([[env*1000+t for env in range(16)] for t in range(128)])
    grouped = chunks(source)
    for chunk in range(8):
        for env in range(16):
            assert grouped[chunk*16+env].tolist() == [env*1000+t for t in range(chunk*16,(chunk+1)*16)]

def test_recurrent_resets_cut_history_and_its_gradients():
    torch.manual_seed(5); model = GardenGRU(); obs = torch.randn(2,8,20,requires_grad=True)
    starts = torch.zeros(2,8,dtype=torch.bool); starts[:,0] = True; starts[:,3] = True
    actual = model.sequence(obs,torch.ones(2,32),starts)
    torch.testing.assert_close(actual[:,:3],model.sequence(obs[:,:3]))
    torch.testing.assert_close(actual[:,3:],model.sequence(obs[:,3:]))
    actual[:,7].sum().backward()
    assert obs.grad[:,:3].abs().sum() == 0 and obs.grad[:,3:].abs().sum() > 0
    prefixes = [obs[0,:3].detach().numpy(), []]
    # Lists are the rollout's owned actual-observation prefixes.
    carried = prefix_state(model,[list(prefixes[0]),[]])
    _,state = model.gru(obs[0,:3].detach()[None])
    torch.testing.assert_close(carried[0],state[0,0]); assert carried[1].abs().sum() == 0

def test_both_actors_receive_real_masked_bc_and_ppo_weight_updates():
    torch.set_num_threads(1)
    splits = json.loads(Path('results/navigation-r1-splits.json').read_text())
    data = demonstrations(splits['demonstration_seeds'])
    assert data[2].sum() == 1290 and len(data[3]) == 256
    for constructor in [GardenGRU,Policy]:
        torch.manual_seed(0); model = constructor(); before = [p.detach().clone() for p in model.parameters()]
        losses = clone(model,data,1)
        assert len(losses) == 6 and all(torch.isfinite(torch.tensor(losses)))
        assert any(not torch.equal(a,b) for a,b in zip(before,model.parameters()))
        before = [p.detach().clone() for p in model.parameters()]
        updates,seeds = ppo(model,0,2048,splits['pool'][:16],lambda *_:None)
        assert updates == 32 and set(seeds) <= set(splits['pool'][:16])
        assert any(not torch.equal(a,b) for a,b in zip(before,model.parameters()))
        assert all(torch.isfinite(p).all() for p in model.parameters())

def test_cycle_rule_includes_collisions_but_allows_an_escape():
    assert cycle([18,19,18,19,18,19])
    assert cycle([18]*6)
    assert not cycle([18,19,18,19,18,20])
    assert not cycle([18,19,18,19,18])
