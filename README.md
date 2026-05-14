# 🐆 HalfCheetah with DQN — Deep Reinforcement Learning
 
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)
![Gymnasium](https://img.shields.io/badge/Gymnasium-MuJoCo-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab%20%7C%20Linux-lightgrey)
 
> Applying Deep Q-Network (DQN) to MuJoCo's `HalfCheetah-v4` continuous control task via action-space discretization, with a head-to-head comparison against PPO.
 
---
 
## 📖 Table of Contents
 
- [Overview](#-overview)
- [Background](#-background)
- [Architecture](#-architecture)
- [Hyperparameters](#-hyperparameters)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Running the Project](#-running-the-project)
- [Results & Metrics](#-results--metrics)
- [PPO vs DQN Comparison](#-ppo-vs-dqn-comparison)
- [Design Decisions & Limitations](#-design-decisions--limitations)
- [Future Work](#-future-work)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)
---
 
## 📌 Overview
 
This project implements a **Deep Q-Network (DQN)** agent to solve the `HalfCheetah-v4` locomotion task from the [Gymnasium MuJoCo](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) benchmark suite. The HalfCheetah environment features a **continuous** 6-dimensional action space, which is inherently incompatible with DQN's discrete action assumption.
 
To bridge this gap, a custom **`DiscretizedActionWrapper`** is introduced that discretizes each action dimension into `bins` uniformly-spaced values, creating a finite combinatorial action grid. The agent is then trained end-to-end using standard DQN with experience replay and a target network.
 
Finally, the DQN agent is **benchmarked against a PPO baseline** on normalized metrics including episodic return, episode length, and training loss — providing a practical illustration of how algorithm-environment fit impacts performance.
 
---
 
## 📚 Background
 
### Environment: `HalfCheetah-v4`
 
| Property | Value |
|---|---|
| Observation Space | `Box(17,)` — joint positions & velocities |
| Action Space | `Box(6,)` — torques for 6 joints, range `[-1, 1]` |
| Reward | Forward velocity minus control cost |
| Episode Termination | After 1,000 steps (no early termination) |
 
### Algorithm: Deep Q-Network (DQN)
 
DQN (Mnih et al., 2015) is an off-policy, value-based RL algorithm that approximates the optimal action-value function Q(s, a) using a deep neural network. Key stabilization techniques include:
 
- **Experience Replay** — stores transitions in a buffer and samples random mini-batches to break temporal correlations
- **Target Network** — a periodically updated copy of the Q-network used to compute stable Bellman targets
- **ε-Greedy Exploration** — balances exploration and exploitation via a decaying epsilon schedule
### Why Discretization?
 
DQN requires a discrete action space. For `HalfCheetah-v4`'s 6-dimensional continuous action space, we apply a **grid discretization**: each dimension is divided into `bins` equally-spaced values. With `bins=3`, this yields `3^6 = 729` discrete actions. While this is a significant approximation, it allows DQN to be applied to this domain without architectural changes.
 
---
 
## 🏗️ Architecture
 
### `DiscretizedActionWrapper`
 
```python
class DiscretizedActionWrapper(gym.ActionWrapper):
    def __init__(self, env, bins=3):
        ...
```
 
Wraps the Gymnasium environment and replaces its continuous `Box` action space with a `Discrete` space of size `bins^n_dims`. Internally stores a lookup table (`discrete_actions`) mapping each integer index to its corresponding continuous action vector, constructed via `np.meshgrid` over per-dimension `np.linspace` grids.
 
---
 
### `ReplayBuffer`
 
```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
```
 
A circular experience replay buffer backed by a `collections.deque`. Supports:
- `push(state, action, reward, next_state, done)` — stores a single transition
- `sample(batch_size)` — returns a random mini-batch as stacked PyTorch tensors
---
 
### `QNetwork`
 
```python
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        ...
```
 
A 3-layer fully connected network:
 
```
Linear(obs_dim → 256) → ReLU
Linear(256 → 256)     → ReLU
Linear(256 → act_dim)
```
 
Outputs a Q-value estimate for every discrete action given an observation.
 
---
 
### `DQNAgent`
 
```python
class DQNAgent:
    def __init__(self, obs_dim, act_dim, device):
        ...
```
 
Orchestrates training with two `QNetwork` instances (online + target):
 
| Component | Detail |
|---|---|
| Optimizer | Adam, `lr=1e-3` |
| Loss Function | Huber Loss (`nn.SmoothL1Loss`) |
| Gradient Clipping | Max norm = 1.0 |
| Exploration | ε-greedy with exponential epsilon decay |
| Target Sync | Hard update every `target_update_freq` steps |
 
**Bellman Update:**
```
target_q = r + γ · (1 − done) · max_a' Q_target(s', a')
loss = HuberLoss(Q_online(s, a), target_q)
```
 
---
 
## ⚙️ Hyperparameters
 
| Parameter | Value | Description |
|---|---|---|
| `max_steps` | 200,000 | Total environment interaction steps |
| `buffer_capacity` | 100,000 | Replay buffer size |
| `batch_size` | 256 | Mini-batch size for gradient updates |
| `gamma` (γ) | 0.99 | Discount factor |
| `epsilon` (initial) | 1.0 | Starting exploration rate |
| `epsilon_decay` | 0.995 | Per-episode multiplicative decay |
| `min_epsilon` | 0.05 | Minimum exploration rate |
| `target_update_freq` | 1,000 | Steps between target network hard updates |
| `bins` | 3 | Discretization bins per action dimension |
| `learning_rate` | 1e-3 | Adam optimizer learning rate |
| `hidden_size` | 256 | Q-network hidden layer size |
| `seed` | 42 | Random seed for reproducibility |
 
---
 
## 📁 Project Structure
 
```
halfcheetah-dqn/
│
├── half_cheetah_with_dqn.py                ← Main training & evaluation script
│
├── dqn_halfcheetah.pt                      ← Saved Q-network weights (post-training)
├── dqn_eval.mp4                            ← Recorded evaluation rollout video
│
├── dqn_halfcheetah_metrics_with_loss.png   ← Raw episode return, length & loss curves
├── dqn_halfcheetah_metrics_normalized.png  ← Normalized versions of the above
├── ppo_vs_dqn_comparison.png               ← Side-by-side PPO vs DQN metric comparison
│
├── returns_norm.npy                        ← PPO normalized episodic returns
├── lengths_norm.npy                        ← PPO normalized episode lengths
├── value_loss_norm.npy                     ← PPO normalized value loss
│
└── README.md                               ← This file
```
 
---
 
## 🚀 Setup & Installation
 
### Prerequisites
 
- Python 3.8+
- pip
- CUDA-compatible GPU (optional but recommended)
- Linux or Google Colab (MuJoCo requires system-level dependencies)
### 1. System Dependencies (Linux / Colab)
 
```bash
apt install python-opengl ffmpeg xvfb -y
pip install pyvirtualdisplay pyglet==1.5.1
```
 
### 2. Python Packages
 
```bash
pip install gymnasium[mujoco] \
            stable-baselines3==2.2.1 \
            torch \
            numpy \
            matplotlib \
            tqdm \
            imageio
```
 
### 3. Virtual Display (Headless Rendering)
 
Required for rendering without a physical monitor (e.g., on a server or Colab):
 
```python
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()
```
 
This is already included at the top of `half_cheetah_with_dqn.py`.
 
---
 
## ▶️ Running the Project
 
### Train the Agent
 
```bash
python half_cheetah_with_dqn.py
```
 
**What happens:**
1. The `HalfCheetah-v4` environment is wrapped with `DiscretizedActionWrapper` (bins=3)
2. DQN training runs for 200,000 environment steps
3. ε decays from 1.0 → 0.05 over training
4. Target network is hard-updated every 1,000 steps
5. Model weights are saved to `dqn_halfcheetah.pt`
6. Training metric plots are generated and saved
### Evaluate the Agent & Record Video
 
Automatically called at the end of training:
 
```python
record_video("HalfCheetah-v4", "dqn_halfcheetah.pt", video_path="dqn_eval.mp4", steps=1000)
```
 
Loads the saved `QNetwork`, runs a greedy rollout for up to 1,000 steps, and saves frames to an MP4 using `imageio`.
 
**View in Colab:**
```python
from IPython.display import HTML
from base64 import b64encode
 
mp4 = open("dqn_eval.mp4", "rb").read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML(f'<video width=640 controls><source src="{data_url}" type="video/mp4"></video>')
```
 
### Generate PPO vs DQN Comparison
 
Ensure the PPO `.npy` files are present in the working directory, then run the comparison block at the bottom of the script. It produces `ppo_vs_dqn_comparison.png` with three side-by-side normalized subplots.
 
---
 
## 📊 Results & Metrics
 
### Training Curves
 
Three metric plots are saved during/after training:
 
#### 1. Raw Metrics — `dqn_halfcheetah_metrics_with_loss.png`
Three subplots:
- **Episode Return** vs. episode number
- **Episode Length** vs. episode number
- **Training Loss** (Huber) vs. gradient update step
#### 2. Normalized Metrics — `dqn_halfcheetah_metrics_normalized.png`
Same three metrics, each min-max normalized to `[0, 1]`:
 
```python
def normalize(array):
    array = np.array(array)
    return (array - array.min()) / (array.max() - array.min() + 1e-8)
```
 
Normalization facilitates cross-algorithm comparison regardless of absolute scale.
 
---
 
## 🆚 PPO vs DQN Comparison
 
The final comparison plot (`ppo_vs_dqn_comparison.png`) overlays both algorithms across three normalized metrics:
 
| Subplot | X-Axis | Y-Axis |
|---|---|---|
| Episodic Return | Training steps | Normalized return |
| Episode Length | Training steps | Normalized length |
| Training Loss | Training steps | Normalized loss |
 
```
PPO  → Blue
DQN  → Orange
```
 
> **Expected outcome:** PPO substantially outperforms DQN on `HalfCheetah-v4`. This is expected — PPO is a policy gradient method natively designed for continuous action spaces, while DQN's grid discretization loses precision and scales poorly with action dimensionality. The comparison is included for educational insight into algorithm-environment fit.
 
---
 
## 🔬 Design Decisions & Limitations
 
### Discretization Trade-offs
- With `bins=3` and 6 actuators, the discrete action space has **729 actions** — covering the continuous space coarsely.
- Increasing `bins` improves precision but causes **combinatorial explosion**: `bins=5` → 15,625 actions, making Q-network outputs very high-dimensional.
- The discretization grid is **uniform and static** — it does not adapt to which regions of action space are actually useful.
### DQN on Continuous Control
- DQN's Q-function maximization (`argmax_a Q(s,a)`) is tractable only when the action space is finite and small.
- For continuous control, methods like **SAC**, **TD3**, and **PPO** are far better suited.
- This implementation serves as an instructive study in the limitations of value-based methods on continuous benchmarks.
### No Separate Evaluation Loop
- Returns are recorded from **training** episodes (with ε-greedy noise), not from greedy evaluation episodes.
- This means reported returns underestimate the true greedy policy performance.
### Replay Buffer Warmup
- Training updates begin as soon as `len(buffer) >= batch_size` — there is no explicit warmup period with random actions only.
---
 
## 🔭 Future Work
 
- [ ] Replace DQN with **SAC** (Soft Actor-Critic) or **TD3** for native continuous-action performance
- [ ] Implement **Dueling DQN** architecture for better value/advantage decomposition
- [ ] Add **Prioritized Experience Replay (PER)** to focus training on high-TD-error transitions
- [ ] Sweep over `bins ∈ {2, 3, 5, 7}` to quantify the precision–performance tradeoff
- [ ] Add a proper **greedy evaluation loop** (every N steps, run K episodes without exploration)
- [ ] Report mean ± std over **multiple random seeds** for statistically robust comparisons
- [ ] Integrate **Weights & Biases** (`wandb`) for experiment tracking and hyperparameter sweeps
- [ ] Extend to additional MuJoCo environments: `Ant-v4`, `Hopper-v4`, `Walker2d-v4`
---
 
## 📄 License
 
This project is released under the [MIT License](https://opensource.org/licenses/MIT).
 
```
MIT License
 
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```
 
---
 
## 🙏 Acknowledgements
 
- [Farama Foundation](https://farama.org/) for maintaining [Gymnasium](https://gymnasium.farama.org/) and the MuJoCo environments
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) for the PPO baseline implementation
- [DeepMind / Google](https://mujoco.org/) for open-sourcing the MuJoCo physics simulator
- **Original DQN Paper:** Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518, 529–533.
- **PPO Paper:** Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
---
 
<p align="center">
  Made with ❤️ using PyTorch, Gymnasium, and MuJoCo
</p>
