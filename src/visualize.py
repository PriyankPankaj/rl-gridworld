import numpy as np
import matplotlib.pyplot as plt
from src.environment import GridWorld

RESULTS_PATH = "results/comparison_results.npy"


def moving_average(data, window=50):
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_training_curves(results, epsilon=1.0, save_path="results/training_curves.png"):
    data = results[epsilon]
    q_smooth = moving_average(data["q_rewards"])
    sarsa_smooth = moving_average(data["sarsa_rewards"])

    plt.figure(figsize=(10, 6))
    plt.plot(q_smooth, label="Q-Learning (off-policy)", linewidth=1.5)
    plt.plot(sarsa_smooth, label="SARSA (on-policy)", linewidth=1.5)
    plt.xlabel("Episode")
    plt.ylabel("Reward (50-episode moving average)")
    plt.title(f"Training Curves: Q-Learning vs SARSA (initial epsilon={epsilon})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()


def plot_convergence_comparison(results, save_path="results/convergence_comparison.png"):
    epsilons = sorted(results.keys())
    q_conv = [results[e]["q_convergence"] or 0 for e in epsilons]
    sarsa_conv = [results[e]["sarsa_convergence"] or 0 for e in epsilons]

    plt.figure(figsize=(9, 6))
    plt.plot(epsilons, q_conv, marker="o", label="Q-Learning")
    plt.plot(epsilons, sarsa_conv, marker="o", label="SARSA")
    plt.xlabel("Initial Epsilon")
    plt.ylabel("Episodes to Convergence")
    plt.title("Convergence Speed Across Full Epsilon Range (0.0–1.0)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

def plot_goal_reach_comparison(results, save_path="results/goal_reach_comparison.png"):
    epsilons = sorted(results.keys())
    q_rates = [results[e]["q_eval"]["goal_reach_rate"] for e in epsilons]
    sarsa_rates = [results[e]["sarsa_eval"]["goal_reach_rate"] for e in epsilons]

    plt.figure(figsize=(9, 6))
    plt.plot(epsilons, q_rates, marker="o", label="Q-Learning")
    plt.plot(epsilons, sarsa_rates, marker="o", label="SARSA")
    plt.xlabel("Initial Epsilon")
    plt.ylabel("Goal-Reaching Rate (%)")
    plt.title("Goal-Reaching Rate Across Full Epsilon Range (0.0–1.0)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()



def plot_policy_grid(agent, env, title, save_path):
    """Visualizes the learned greedy policy as arrows on the grid."""
    arrow_map = {0: "^", 1: "v", 2: "<", 3: ">"}
    fig, ax = plt.subplots(figsize=(6, 6))

    for r in range(env.size):
        for c in range(env.size):
            pos = (r, c)
            state_idx = r * env.size + c

            if pos == env.goal:
                ax.text(c, r, "G", ha="center", va="center", fontsize=16, color="green", fontweight="bold")
            elif pos in env.hazards:
                ax.text(c, r, "X", ha="center", va="center", fontsize=16, color="red", fontweight="bold")
            else:
                best_action = np.argmax(agent.q_table[state_idx])
                ax.text(c, r, arrow_map[best_action], ha="center", va="center", fontsize=14)

    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(env.size - 0.5, -0.5)
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.grid(True)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    results = np.load(RESULTS_PATH, allow_pickle=True).item()

    plot_training_curves(results, epsilon=1.0)
    plot_convergence_comparison(results)
    plot_goal_reach_comparison(results)

    # Retrain once more at epsilon=1.0 just to get agents for the policy grid
    # (compare.py doesn't currently save the trained agent objects, only reward histories)
    from src.compare import train_with_tracking
    from src.q_learning import QLearningAgent
    from src.sarsa import SARSAAgent

    q_agent, _ = train_with_tracking(QLearningAgent, GridWorld(), epsilon=1.0)
    sarsa_agent, _ = train_with_tracking(SARSAAgent, GridWorld(), epsilon=1.0)

    plot_policy_grid(q_agent, GridWorld(), "Q-Learning Learned Policy", "results/q_learning_policy.png")
    plot_policy_grid(sarsa_agent, GridWorld(), "SARSA Learned Policy", "results/sarsa_policy.png")