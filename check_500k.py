import json, os
import matplotlib.pyplot as plt
from env_battery_continuous import ContinuousNavBatteryEnv
from lagrangian_ppo import train_lagrangian_ppo, _quick_rollout_for_lambda

HP_PATH = 'results/hyperparams_lagrangian_battery.json'
hp = json.load(open(HP_PATH))
eta = hp.get('lagrangian_eta', 0.1)
d_threshold = hp.get('lagrangian_d_threshold', 0.05)
print(f'Using tuned hp: {hp}')

model, history = train_lagrangian_ppo(
    ContinuousNavBatteryEnv, cost_key='crashed', signal_key='r_battery',
    net_size=[64], timesteps=500_000, seed=0, hp=hp,
    adversarial_weight=8, d_threshold=d_threshold, eta=eta, n_updates=20,
    return_history=True,
)
for row in history:
    print(f"  steps={row['steps']:>7} cost_rate={row['cost_rate']:.3f} success_rate={row['success_rate']:.3f} lambda={row['lambda']:.4f}")

final = _quick_rollout_for_lambda(model, lambda: ContinuousNavBatteryEnv(adversarial_weight=8), cost_key='crashed', n_episodes=100, seed=999000)
print(f"\nFinal (100 episodes, override_w8): success_rate={final['success_rate']:.3f} cost_rate={final['cost_rate']:.3f} (target {d_threshold})")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
steps = [r['steps'] for r in history]
axes[0].plot(steps, [r['success_rate'] for r in history], marker='o'); axes[0].set_title('success_rate'); axes[0].set_xlabel('steps')
axes[1].plot(steps, [r['cost_rate'] for r in history], marker='o', color='tab:red'); axes[1].axhline(d_threshold, color='gray', linestyle='--'); axes[1].set_title('cost_rate (target=0.05)'); axes[1].set_xlabel('steps')
axes[2].plot(steps, [r['lambda'] for r in history], marker='o', color='tab:purple'); axes[2].set_title('lambda'); axes[2].set_xlabel('steps')
fig.tight_layout()
os.makedirs('results', exist_ok=True)
fig.savefig('results/sanity_check_lagrangian_battery_500k.png', dpi=150)
print('Saved results/sanity_check_lagrangian_battery_500k.png')
