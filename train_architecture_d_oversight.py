"""
train_architecture_d_oversight.py
------------------------------------
Thin per-experiment wrapper around lagrangian_ppo.py's shared
train_lagrangian_ppo(), fixing env_ctor=ContinuousNavOversightEnv and
cost_key="violated" -- these are the only two things that differ from
train_architecture_d_battery.py.

Import this alongside train_oversight.py's existing
train_architecture_a/train_g/train_rho/train_gate in
run_experiment_oversight.py: Architecture D needs none of B/C's
arbitration machinery (it's single-loop, like A), so nothing else
about the existing pipeline changes.
"""

from env_oversight import ContinuousNavOversightEnv
from lagrangian_ppo import train_lagrangian_ppo


def train_architecture_d(net_size, timesteps, seed, hp,
                          adversarial_weight=0.0, d_threshold=0.05,
                          eta=None, n_updates=None, return_history=False):
    """
    Architecture D for the oversight signal: Lagrangian-PPO,
    replacing env_oversight.py's fixed -50.0 terminal violation
    penalty with the learned Lagrange multiplier, while KEEPING the
    environment's own dense, non-terminal shaping (-speed while
    oversight active) unchanged -- see lagrangian_ppo.py's
    LagrangianRewardWrapper docstring for why discarding that dense
    term (an earlier version of this code did) reproduces the
    project's own previously-diagnosed sparse-gradient problem.

    eta=None falls back to hp["lagrangian_eta"] -- pass the TUNED value
    from tune_lagrangian_eta_oversight.py's output for the real sweep,
    not this fallback, per this project's standing discipline.
    """
    return train_lagrangian_ppo(
        ContinuousNavOversightEnv, cost_key="violated", signal_key="r_oversight",
        net_size=net_size, timesteps=timesteps, seed=seed, hp=hp,
        adversarial_weight=adversarial_weight,
        d_threshold=d_threshold, eta=eta, n_updates=n_updates,
        return_history=return_history,
    )


if __name__ == "__main__":
    # Smoke test -- mirrors train_oversight.py's own __main__ pattern.
    #   python train_architecture_d_oversight.py
    hp = {"learning_rate": 3e-4, "gamma": 0.99, "log_std_init": -0.5, "lagrangian_eta": 0.1}
    model, history = train_architecture_d(net_size=[16], timesteps=4_000, seed=0, hp=hp,
                                           return_history=True)
    print("Architecture D (oversight) smoke test passed. History:")
    for row in history:
        print(f"  steps={row['steps']:>6} violation_rate~={row['cost_rate']:.3f} "
              f"success_rate={row['success_rate']:.3f} lambda={row['lambda']:.4f}")
