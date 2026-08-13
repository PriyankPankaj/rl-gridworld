import numpy as np
from src.environment import GridWorld
from src.q_learning import QLearningAgent
from src.sarsa import SARSAAgent
from src.evaluate import evaluate_agent

N_EPISODES = 5000
MAX_STEPS = 200
EPSILON_VALUES = [round(x * 0.1, 1) for x in range(0, 11)]  # 0.0, 0.1, 0.2, ..., 1.0 


def train_with_tracking(agent_cls, env, n_episodes=N_EPISODES, epsilon=1.0):
    agent = agent_cls(env.n_states, env.n_actions, epsilon=epsilon)
    rewards_per_episode = []

    is_sarsa = agent_cls.__name__ == "SARSAAgent"

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0

        if is_sarsa:
            action = agent.select_action(state)
            for _ in range(MAX_STEPS):
                next_state, reward, done = env.step(action)
                next_action = agent.select_action(next_state)
                agent.update(state, action, reward, next_state, next_action, done)
                state, action = next_state, next_action
                total_reward += reward
                if done:
                    break
        else:
            for _ in range(MAX_STEPS):
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break

        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)

    return agent, rewards_per_episode


def find_convergence_episode(rewards, window=100, threshold=-12.0):
    """Finds the first episode where the rolling average reward stabilizes
    above a threshold (a simple, defensible convergence definition)."""
    for i in range(window, len(rewards)):
        if np.mean(rewards[i - window:i]) >= threshold:
            return i
    return None


if __name__ == "__main__":
    results = {}

    for eps in EPSILON_VALUES:
        print(f"\n=== Starting epsilon = {eps} ===")

        q_agent, q_rewards = train_with_tracking(QLearningAgent, GridWorld(), epsilon=eps)
        sarsa_agent, sarsa_rewards = train_with_tracking(SARSAAgent, GridWorld(), epsilon=eps)

        q_eval = evaluate_agent(q_agent, GridWorld())
        sarsa_eval = evaluate_agent(sarsa_agent, GridWorld())

        q_convergence = find_convergence_episode(q_rewards)
        sarsa_convergence = find_convergence_episode(sarsa_rewards)

        print(f"Q-Learning  -> goal_reach: {q_eval['goal_reach_rate']:.1f}%  "
              f"convergence_episode: {q_convergence}")
        print(f"SARSA       -> goal_reach: {sarsa_eval['goal_reach_rate']:.1f}%  "
              f"convergence_episode: {sarsa_convergence}")

        results[eps] = {
            "q_rewards": q_rewards,
            "sarsa_rewards": sarsa_rewards,
            "q_eval": q_eval,
            "sarsa_eval": sarsa_eval,
            "q_convergence": q_convergence,
            "sarsa_convergence": sarsa_convergence,
        }

    np.save("results/comparison_results.npy", results, allow_pickle=True)
    print("\nSaved results to results/comparison_results.npy")