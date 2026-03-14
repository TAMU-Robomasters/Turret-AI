import torch
import numpy as np
from turret_env import TurretEnv
from turret_model_train import PolicyGRU

def test_policy(model_path="best_policy_gru.pt", n_episodes=10, render=True):
    # Use the same obs_dim, action_dim and hidden_dim as in training
    obs_dim = 7
    action_dim = 3
    hidden_dim = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = TurretEnv()
    policy = PolicyGRU(obs_dim, hidden_dim, action_dim).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()

    episode_returns = []
    for ep in range(n_episodes):
        obs = env.reset()
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        obs = obs.unsqueeze(0).unsqueeze(0)
        hidden = None
        done = False
        ep_return = 0.0
        step = 0

        while not done:
            with torch.no_grad():
                action, hidden = policy.mean_action(obs, hidden)
            action_np = action.squeeze(0).squeeze(0).cpu().numpy()
            obs_next, reward, done, info = env.step(action_np)
            ep_return += reward
            step += 1
            obs = torch.as_tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            if hasattr(env, "render") and render:
                env.render()
        print(f"Episode {ep + 1}: Return = {ep_return:.2f}, Steps = {step}")
        episode_returns.append(ep_return)
    print(f"Average Return over {n_episodes} episodes: {np.mean(episode_returns):.2f}")

if __name__ == "__main__":
    test_policy()
