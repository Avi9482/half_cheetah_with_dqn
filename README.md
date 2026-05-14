# 🐆 HalfCheetah with DQN — Deep Reinforcement Learning
 
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)
![Gymnasium](https://img.shields.io/badge/Gymnasium-MuJoCo-green)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab%20%7C%20Linux-lightgrey)
 
> Applying Deep Q-Network (DQN) to MuJoCo's `HalfCheetah-v4` continuous control task via action-space discretization, with a head-to-head comparison against PPO.
 
---
 
 
## 📌 Overview
 
This project implements a **Deep Q-Network (DQN)** agent to solve the `HalfCheetah-v4` locomotion task from the [Gymnasium MuJoCo](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) benchmark suite. The HalfCheetah environment features a **continuous** 6-dimensional action space, which is inherently incompatible with DQN's discrete action assumption.
 
To bridge this gap, a custom **`DiscretizedActionWrapper`** is introduced that discretizes each action dimension into `bins` uniformly-spaced values, creating a finite combinatorial action grid. The agent is then trained end-to-end using standard DQN with experience replay and a target network.
 
 
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
 

 
Wraps the Gymnasium environment and replaces its continuous `Box` action space with a `Discrete` space of size `bins^n_dims`. Internally stores a lookup table (`discrete_actions`) mapping each integer index to its corresponding continuous action vector, constructed via `np.meshgrid` over per-dimension `np.linspace` grids.
 
---
 
### `ReplayBuffer`
 
 
A circular experience replay buffer backed by a `collections.deque`. Supports:
- `push(state, action, reward, next_state, done)` — stores a single transition
- `sample(batch_size)` — returns a random mini-batch as stacked PyTorch tensors
---
 
### `QNetwork`
 
 
A 3-layer fully connected network:
 
 
Outputs a Q-value estimate for every discrete action given an observation.
 
---
 
### `DQNAgent`
 

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
│                   ← PPO normalized value loss
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
 
 

 
 

 

