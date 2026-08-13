import numpy as np
from src.environment import GridWorld
from src.train import train_q_learning, train_sarsa

N_EVAL_EPISODES = 200


def evaluate_agent(agent, env, n_episodes=N_EVAL_EPISODES, max_steps=100):
    """Runs the trained agent greedily (no exploration) and measures how
    often it reaches the goal vs. hits a hazard or times out."""
    goals_reached = 0
    hazards_hit = 0
    timeouts = 0
    steps_to_goal = []

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # fully greedy for evaluation

    for _ in range(n_episodes):
        state = env.reset()
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            if done:
                if reward == 10.0:
                    goals_reached += 1
                    steps_to_goal.append(step + 1)
                elif reward == -10.0:
                    hazards_hit += 1
                break
        else:
            timeouts += 1

    agent.epsilon = old_epsilon  # restore

    return {
        "goal_reach_rate": goals_reached / n_episodes * 100,
        "hazard_hit_rate": hazards_hit / n_episodes * 100,
        "timeout_rate": timeouts / n_episodes * 100,
        "avg_steps_to_goal": np.mean(steps_to_goal) if steps_to_goal else None,
    }


if __name__ == "__main__":
    print("Training Q-Learning...")
    env_q = GridWorld()
    q_agent, q_rewards = train_q_learning(env_q)

    print("Training SARSA...")
    env_sarsa = GridWorld()
    sarsa_agent, sarsa_rewards = train_sarsa(env_sarsa)

    print("\n--- Evaluation (200 episodes, greedy policy) ---\n")

    q_results = evaluate_agent(q_agent, GridWorld())
    print("Q-Learning:")
    for k, v in q_results.items():
        print(f"  {k}: {v:.2f}" if v is not None else f"  {k}: N/A")

    sarsa_results = evaluate_agent(sarsa_agent, GridWorld())
    print("\nSARSA:")
    for k, v in sarsa_results.items():
        print(f"  {k}: {v:.2f}" if v is not None else f"  {k}: N/A")