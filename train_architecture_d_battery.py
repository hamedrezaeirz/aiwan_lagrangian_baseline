"""
train_architecture_d_battery.py
-----------------------------------
Thin per-experiment wrapper around lagrangian_ppo.py's shared
train_lagrangian_ppo(), fixing env_ctor=ContinuousNavBatteryEnv and
cost_key="crashed" -- the direct battery-signal analogue of
train_architecture_d_oversight.py. Built second, per the confirmed
priority order (oversight first).
"""

from env_battery_continuous import ContinuousNavBatteryEnv
from lagrangian_ppo import train_lagrangian_ppo


def train_architecture_d(net_size, timesteps, seed, hp,
                          adversarial_weight=0.0, d_threshold=0.05,
                          eta=None, n_updates=None, return_history=False):
    """
    Architecture D for the battery signal: Lagrangian-PPO, replacing
    env_battery_continuous.py's fixed -50.0 terminal crash penalty
    with the learned Lagrange multiplier, while KEEPING the
    environment's own dense, non-terminal shaping (the urgency/charger-
    distance term in r_battery) unchanged. Mirrors
    train_architecture_d_oversight.py exactly, with cost_key="crashed"
    and signal_key="r_battery" in place of "violated"/"r_oversight".

    eta=None falls back to hp["lagrangian_eta"] -- pass the TUNED value
    from tune_lagrangian_eta_battery.py's output (not yet built --
    mirror tune_lagrangian_eta_oversight.py with the env/keys swapped,
    once oversight's Architecture D sweep is running) for the real
    sweep, not this fallback.
    """
    return train_lagrangian_ppo(
        ContinuousNavBatteryEnv, cost_key="crashed", signal_key="r_battery",
        net_size=net_size, timesteps=timesteps, seed=seed, hp=hp,
        adversarial_weight=adversarial_weight,
        d_threshold=d_threshold, eta=eta, n_updates=n_updates,
        return_history=return_history,
    )


if __name__ == "__main__":
    # Smoke test -- mirrors train_battery.py's own __main__ pattern.
    #   python train_architecture_d_battery.py
    hp = {"learning_rate": 3e-4, "gamma": 0.99, "log_std_init": -0.5, "lagrangian_eta": 0.1}
    model, history = train_architecture_d(net_size=[16], timesteps=4_000, seed=0, hp=hp,
                                           return_history=True)
    print("Architecture D (battery) smoke test passed. History:")
    for row in history:
        print(f"  steps={row['steps']:>6} crash_rate~={row['cost_rate']:.3f} "
              f"success_rate={row['success_rate']:.3f} lambda={row['lambda']:.4f}")
