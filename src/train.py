import numpy as np
from src.environment import GridWorld
from src.q_learning import QLearningAgent
from src.sarsa import SARSAAgent

N_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 200


def train_q_learning(env, n_episodes=N_EPISODES):
    agent = QLearningAgent(env.n_states, env.n_actions)
    episode_rewards = []

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0

        for _ in range(MAX_STEPS_PER_EPISODE):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

    return agent, episode_rewards


def train_sarsa(env, n_episodes=N_EPISODES):
    agent = SARSAAgent(env.n_states, env.n_actions)
    episode_rewards = []

    for ep in range(n_episodes):
        state = env.reset()
        action = agent.select_action(state)
        total_reward = 0

        for _ in range(MAX_STEPS_PER_EPISODE):
            next_state, reward, done = env.step(action)
            next_action = agent.select_action(next_state)
            agent.update(state, action, reward, next_state, next_action, done)
            state, action = next_state, next_action
            total_reward += reward
            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

    return agent, episode_rewards


if __name__ == "__main__":
    print("Training Q-Learning...")
    env_q = GridWorld()
    q_agent, q_rewards = train_q_learning(env_q)
    print(f"Q-Learning: avg reward (last 100 eps): {np.mean(q_rewards[-100:]):.2f}")

    print("\nTraining SARSA...")
    env_sarsa = GridWorld()
    sarsa_agent, sarsa_rewards = train_sarsa(env_sarsa)
    print(f"SARSA: avg reward (last 100 eps): {np.mean(sarsa_rewards[-100:]):.2f}")