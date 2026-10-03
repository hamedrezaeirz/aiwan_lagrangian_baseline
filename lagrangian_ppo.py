"""
lagrangian_ppo.py
--------------------
Shared core for Architecture D (Experiment 3): a Lagrangian-PPO
constrained-RL baseline, replacing Architecture A's simple linear
reward-shaping (R = r_goal + lambda_fixed * r_signal) with a proper
primal-dual method (R = r_goal - lambda_t * cost, lambda_t updated by
dual ascent toward a target constraint violation rate).

Built ONCE, here, and reused by both experiments (see
train_architecture_d_oversight.py / train_architecture_d_battery.py,
thin per-experiment wrappers) rather than duplicated -- this is safe
because env_oversight.py and env_battery_continuous.py share the same
underlying continuous-navigation dynamics (env_battery_continuous.py's
own docstring: "reusing the exact same movement dynamics ... means the
ONLY thing that differs between the two experiments is the
structurally-separate signal itself"). The only per-experiment
difference Lagrangian-PPO needs is WHICH info key holds the binary
cost signal ("violated" for oversight, "crashed" for battery) -- this
is threaded through as `cost_key`, not hardcoded here.

Design (confirmed decision, see this project's conversation history):
  R_train = r_goal - lambda_t * cost,  cost = 1 if constraint violated
            this step else 0 (env's own info[cost_key])
  lambda_{t+1} = max(0, lambda_t + eta * (measured_cost_rate - d_threshold))

d_threshold = 0.05 (a small positive target, not exactly 0 -- see
module docstring rationale: a target of exactly 0 risks numerical
divergence of lambda, and a small positive tolerance is also more
realistic for what a real safety system would target).

eta (the dual-ascent learning rate) is a genuine hyperparameter that
needs tuning per experiment -- see tune_lagrangian_eta_oversight.py
(this project's own established discipline: nothing chosen after
looking at final results). It is NOT hardcoded to a single value here;
callers pass it explicitly (from a tuned hyperparams file) or accept
this module's conservative default.

Architecture D is single-loop, like A -- no gate, no arbitration class
needed. Its trained model has the exact same (obs) -> action interface
as any other SB3 model, so it plugs directly into the EXISTING
evaluate_cell/rollout/as_policy_fn pipeline (evaluate_oversight.py /
evaluate_battery.py) with NO changes needed there.
"""

import numpy as np
from stable_baselines3 import PPO
import gymnasium as gym


class MutableLambda:
    """A tiny mutable box holding the current Lagrange multiplier, so
    LagrangianRewardWrapper can see lambda updates made BETWEEN
    training chunks without the environment/model being rebuilt each
    time (rebuilding would reset the policy's optimizer state, which
    we don't want)."""

    def __init__(self, value=0.0):
        self.value = float(value)


class LagrangianRewardWrapper(gym.Wrapper):
    """
    Generic Lagrangian reward wrapper: R = r_goal + shaped_signal, where
    shaped_signal REPLACES the environment's own fixed terminal penalty
    with the learned Lagrange multiplier, but KEEPS the environment's
    own dense, non-terminal shaping term unchanged.

    Why this matters (bug found and fixed during this project's own
    validation -- see conversation history): both env_oversight.py's
    info["r_oversight"] and env_battery_continuous.py's info["r_battery"]
    are ALREADY structured as (fixed_large_penalty if cost else
    dense_continuous_shaping) -- e.g. oversight's
    `-50.0 if violated else (-1.0*speed if already_active else 0.0)`.
    That dense shaping term (discouraging risky speed WHILE the signal
    is active, well before any actual violation) is exactly what lets
    Architecture A learn at all -- it is NOT redundant with the sparse
    binary cost. An earlier version of this wrapper used ONLY
    `cost = info[cost_key]` and threw the dense term away entirely,
    reproducing the exact sparse-gradient problem this project already
    diagnosed and fixed once for rho_oversight (diagnose_rho_instability.py):
    a validation tuning run showed cost_rate stuck around 0.46-0.49
    regardless of eta spanning 0.01-1.0, a 100x range, which should be
    impossible if the constraint mechanism were actually shaping
    behavior -- the tell that lambda's magnitude wasn't the bottleneck,
    gradient density was.

    Fix: only the FIXED terminal penalty is replaced by -lambda_t (this
    is the actual Lagrangian relaxation -- letting dual ascent find the
    right penalty size instead of a hand-picked -50.0); the dense,
    non-terminal shaping term the environment already provides is kept
    exactly as every other architecture already benefits from it.

    signal_key selects which info field holds this (fixed_penalty if
    cost else dense_shaping) structure for this experiment
    ("r_oversight" for oversight, "r_battery" for battery) -- both
    environments already expose it, so no environment changes are
    needed.
    """

    def __init__(self, env, cost_key, signal_key, lambda_container):
        super().__init__(env)
        self.cost_key = cost_key
        self.signal_key = signal_key
        self.lambda_container = lambda_container

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        cost = float(info[self.cost_key])
        if cost > 0:
            # Replace the environment's fixed terminal penalty with the
            # learned Lagrange multiplier -- the actual relaxation.
            shaped_signal = -self.lambda_container.value
        else:
            # Keep the environment's own dense, non-terminal shaping
            # UNCHANGED -- this is the gradient signal that makes the
            # constraint learnable at all before any violation occurs.
            shaped_signal = info[self.signal_key]
        reward = info["r_goal"] + shaped_signal
        info["cost"] = cost
        return obs, reward, terminated, truncated, info


def _make_ppo(env, net_arch, seed, hp):
    """Same PPO settings pattern as train_oversight.py/train_battery.py's
    own _make_ppo -- duplicated here deliberately (not imported from
    either), matching this project's existing convention of small,
    self-contained per-purpose _make_ppo copies rather than a shared
    import that would couple this module to one experiment's package."""
    return PPO(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": net_arch, "log_std_init": hp.get("log_std_init", -0.5)},
        learning_rate=hp.get("learning_rate", 3e-4),
        gamma=hp.get("gamma", 0.99),
        seed=seed,
        verbose=0,
    )


def _quick_rollout_for_lambda(model, base_env_fn, cost_key, n_episodes, seed):
    """
    Measures cost_rate (fraction of EPISODES with at least one
    violation/crash -- same per-episode definition evaluate_*.py's
    rollout() uses for violation_rate/crash_rate, so the constraint
    threshold d_threshold is on the SAME scale reported everywhere
    else in this project) and success_rate, on the UNWRAPPED base env
    (not the Lagrangian-reward-wrapped one -- reward doesn't matter
    here, only the info flags do). Used internally between training
    chunks to update lambda; NOT a substitute for the real
    evaluate_cell() call the final sweep uses for reporting.
    """
    env = base_env_fn()
    successes = costs = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        info = {}
        for _ in range(env.max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        successes += int(info.get("reached_goal", False))
        costs += int(info.get(cost_key, False))
    return {
        "success_rate": successes / n_episodes,
        "cost_rate": costs / n_episodes,
    }


def train_lagrangian_ppo(env_ctor, cost_key, signal_key, net_size, timesteps, seed, hp,
                          adversarial_weight=0.0, d_threshold=0.05, eta=None,
                          update_interval=10_000, n_updates=None,
                          n_eval_episodes_for_lambda=30,
                          lambda_init=None, return_history=False):
    """
    Trains Architecture D (Lagrangian-PPO) for `timesteps` total steps,
    updating the Lagrange multiplier every `update_interval` steps via
    dual ascent toward d_threshold.

    env_ctor: the environment CLASS (ContinuousNavOversightEnv or
        ContinuousNavBatteryEnv), not an instance -- called internally
        as env_ctor(adversarial_weight=adversarial_weight).
    cost_key: "violated" (oversight) or "crashed" (battery) -- the
        info field read as the binary per-step constraint-violation
        signal.
    signal_key: "r_oversight" (oversight) or "r_battery" (battery) --
        the info field holding the environment's own (fixed_penalty if
        cost else dense_shaping) structure. See
        LagrangianRewardWrapper's docstring for why this must NOT be
        discarded in favor of cost_key alone.
    update_interval / n_updates: how often lambda is updated. With
        n_updates=None (default), lambda updates every ~update_interval
        steps (rounded to whole PPO iterations). With n_updates=N, the
        run is split into ~N equal groups of PPO iterations instead, so
        EVERY budget gets the same number of dual-ascent updates --
        essential for the real sweep, where a fixed 10,000-step interval
        would give a 10,000-step budget only ONE update (after training
        has already finished, i.e. lambda would stay 0 the whole time).
        The tuning/sanity-check scripts use the default (~20 updates at
        200k), so a sweep run with n_updates=20 matches what eta was
        tuned against.
    eta: dual-ascent learning rate. If None, falls back to
        hp.get("lagrangian_eta", 0.1) -- but this fallback is a
        placeholder, not a validated default; use
        tune_lagrangian_eta_oversight.py's output for the real sweep,
        per this project's standing discipline against untuned
        hyperparameters in real results.
    lambda_init: starting value for lambda. If None, falls back to
        hp.get("lagrangian_lambda_init", 0.0) -- starting at 0 means
        the terminal-violation penalty starts at 0 (not the
        environment's own -50), while the dense non-terminal shaping
        term is present from step one regardless of lambda -- so the
        agent is never entirely without gradient toward compliance,
        even before the first lambda update fires.
    return_history: if True, also returns the list of
        {steps, cost_rate, lambda} dicts recorded at every update --
        use this for sanity_check_lagrangian_*.py's convergence check,
        not needed for the real sweep's normal evaluate_cell() path.

    Returns: the trained SB3 model (same (obs) -> action interface as
    any other model in this project -- plugs directly into
    evaluate_cell/as_policy_fn with no changes needed there), and
    optionally the training history.
    """
    eta = eta if eta is not None else hp.get("lagrangian_eta", 0.1)
    lam = MutableLambda(lambda_init if lambda_init is not None else hp.get("lagrangian_lambda_init", 0.0))

    base_env_fn = lambda: env_ctor(adversarial_weight=adversarial_weight)
    env = LagrangianRewardWrapper(base_env_fn(), cost_key=cost_key, signal_key=signal_key,
                                   lambda_container=lam)
    model = _make_ppo(env, net_size, seed, hp)

    history = []
    eval_seed_base = 90_000 + seed * 1000  # disjoint block per seed, well clear of
                                             # the real sweep's own eval seeds

    # Chunks are defined in whole PPO ITERATIONS (multiples of model.n_steps,
    # SB3's rollout length, 2048 by default), not raw timesteps: SB3's
    # PPO.learn() always rounds a request UP to a whole number of rollouts,
    # so requesting e.g. 10,000 steps in 50 separate calls silently runs
    # ~512,000 steps in total, while Architecture A's single
    # learn(500_000) call runs ~501,760. Counting in iterations makes D's
    # total training volume IDENTICAL to A's for the same nominal budget --
    # needed for a fair paired comparison.
    n_steps = model.n_steps
    total_iters = -(-timesteps // n_steps)  # ceil
    if n_updates is not None:
        iters_per_update = max(1, -(-total_iters // n_updates))
    else:
        iters_per_update = max(1, round(update_interval / n_steps))

    iters_done = 0
    while iters_done < total_iters:
        group = min(iters_per_update, total_iters - iters_done)
        model.learn(total_timesteps=group * n_steps, reset_num_timesteps=False)
        iters_done += group

        metrics = _quick_rollout_for_lambda(
            model, base_env_fn, cost_key,
            n_episodes=n_eval_episodes_for_lambda,
            seed=eval_seed_base + iters_done,
        )
        lam.value = max(0.0, lam.value + eta * (metrics["cost_rate"] - d_threshold))

        history.append({
            "steps": model.num_timesteps,
            "cost_rate": metrics["cost_rate"],
            "success_rate": metrics["success_rate"],
            "lambda": lam.value,
        })

    if return_history:
        return model, history
    return model


if __name__ == "__main__":
    # Quick manual smoke test -- verifies the chunked training loop and
    # lambda dual-ascent update run end to end, on the SMALLEST
    # possible budget, using the oversight environment as the concrete
    # example (battery would work identically, just swap env_ctor/
    # cost_key). Does NOT confirm good convergence at a real budget --
    # that's sanity_check_lagrangian_oversight.py's job.
    #   python lagrangian_ppo.py
    from env_oversight import ContinuousNavOversightEnv

    hp = {"learning_rate": 3e-4, "gamma": 0.99, "log_std_init": -0.5}
    model, history = train_lagrangian_ppo(
        ContinuousNavOversightEnv, cost_key="violated", signal_key="r_oversight",
        net_size=[16], timesteps=4_000, seed=0, hp=hp,
        d_threshold=0.05, eta=0.1, update_interval=2_000,
        n_eval_episodes_for_lambda=10, return_history=True,
    )
    print("Smoke test passed -- trained without error. History:")
    for row in history:
        print(f"  steps={row['steps']:>6} cost_rate={row['cost_rate']:.3f} "
              f"success_rate={row['success_rate']:.3f} lambda={row['lambda']:.4f}")
