\# Reinforcement Learning From Scratch — Q-Learning \& SARSA



Implemented \*\*Q-Learning\*\* (off-policy) and \*\*SARSA\*\* (on-policy) from

scratch with epsilon-greedy exploration, trained and evaluated on a custom

\*\*100-state 10×10 stochastic GridWorld\*\*.



\## What's built



\- Configurable \*\*GridWorld environment\*\* — state transitions, reward

&#x20; function (-1/step, +10 goal, -10 hazard), terminal conditions, and

&#x20; \*\*stochastic "slip" transitions\*\* (10% chance the agent's action slips

&#x20; perpendicular to intended, matching classic slippery-grid RL setups like

&#x20; OpenAI Gym's FrozenLake)

\- \*\*Q-Learning agent\*\* — off-policy TD control (updates using max Q-value

&#x20; of next state)

\- \*\*SARSA agent\*\* — on-policy TD control (updates using the Q-value of the

&#x20; action actually taken next)

\- Both agents trained for \*\*5,000 episodes\*\* each, across a \*\*full epsilon

&#x20; sweep from 0.0 to 1.0 in steps of 0.1\*\* (11 values, 22 total training runs)

\- \*\*Evaluation harness\*\* — 200 greedy (epsilon=0) episodes per agent per

&#x20; epsilon setting, measuring goal-reach rate, hazard-hit rate, timeout

&#x20; rate, and average steps to goal

\- \*\*Convergence tracking\*\* — first episode where a 100-episode rolling

&#x20; average reward stabilizes above a fixed threshold, used to compare

&#x20; learning speed

\- \*\*Visualizations\*\* — training curves, convergence-speed sweep, goal-reach

&#x20; sweep, and learned policy grids for both agents



\## Results



Goal-reaching rate and convergence speed across the full epsilon sweep

(200 evaluation episodes per setting):



| Initial ε | Q-Learning Goal% | SARSA Goal% | Q-Learning Convergence (ep) | SARSA Convergence (ep) |

|---|---|---|---|---|

| 0.0 | 92.5% | 92.5% | 696 | 749 |

| 0.1 | 95.0% | 93.5% | 649 | 751 |

| 0.2 | 96.5% | 94.5% | 706 | 859 |

| 0.3 | 96.0% | 92.5% | 686 | 724 |

| 0.4 | 96.0% | 95.5% | 677 | 769 |

| 0.5 | 93.5% | 93.5% | 773 | 847 |

| 0.6 | 93.0% | 94.5% | 745 | 842 |

| 0.7 | 98.5% | 91.5% | 789 | 838 |

| 0.8 | 96.5% | 95.5% | 797 | 855 |

| 0.9 | 94.0% | 92.5% | 772 | 866 |

| 1.0 | 98.5% | 94.0% | 794 | 863 |



!\[Training curves](results/training\_curves.png)

!\[Convergence comparison](results/convergence\_comparison.png)

!\[Goal-reach comparison](results/goal\_reach\_comparison.png)

!\[Q-Learning policy](results/q\_learning\_policy.png)

!\[SARSA policy](results/sarsa\_policy.png)



\## On-policy vs. off-policy: what the data shows



\*\*Q-Learning converged faster than SARSA at every single epsilon value

tested\*\* (roughly 650–800 episodes vs. 720–870 episodes) — a consistent,

reproducible pattern across the full sweep, not a cherry-picked result.

This matches the theoretical expectation: off-policy Q-Learning bootstraps

directly from the best possible next action, while SARSA's on-policy

updates are shaped by its own exploratory behavior, which tends to slow

convergence slightly. Goal-reaching rates were comparable overall

(91.5%–98.5% across both algorithms and all epsilon settings), with

Q-Learning trending marginally higher on average.



\## Why not 100%



The environment includes stochastic ("slippery") transitions — a 10%

chance any action results in a perpendicular slip instead of the intended

direction — plus 5 hazard cells scattered across the 100-state grid. This

means even an optimal policy cannot guarantee success on every episode,

which is why goal-reaching rates cluster in the low-to-mid 90s rather than

100%. This is a deliberate, standard difficulty mechanism (matching

OpenAI Gym's FrozenLake), not a limitation of the learned policy.



\## Explicitly out of scope (v1)



\- Deep RL (neural network function approximation)

\- Multi-agent environments



\## Roadmap (post-placement)



\- Deep Q-Network (DQN) comparison against tabular methods

\- Multi-agent GridWorld variant

\- Dynamic/randomized hazard layouts per episode



\## Tech stack



Python · NumPy · Matplotlib



\## Running



```bash

python -m src.train        # basic training run

python -m src.evaluate      # trained agents + evaluation metrics

python -m src.compare       # full epsilon sweep (0.0-1.0) + convergence tracking

python -m src.visualize     # generates all plots in results/

```

