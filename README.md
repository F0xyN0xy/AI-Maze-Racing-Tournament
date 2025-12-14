# 🏁 AI Maze Racing Tournament

An interactive visualization of reinforcement learning where AI agents learn to navigate mazes using Q-Learning, then compete in a head-to-head race!

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Pygame](https://img.shields.io/badge/pygame-2.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-purple.svg)

## 🎮 Features

- **Live Training Visualization**: Watch 4 AI agents learn to navigate mazes in real-time
- **Q-Learning Algorithm**: Reinforcement learning with reward shaping and exploration strategies
- **Competitive Racing**: After training, AIs race to find the optimal path
- **Modern UI**: Smooth scrolling sidebar with detailed statistics and metrics
- **Resizable Window**: Fully scalable interface that adapts to any screen size
- **Performance Metrics**: Track goals reached, exploration progress, rewards, and path efficiency

## 🚀 Quick Start

### Create Virtual Environment

1. Select Python 3.12.0 (the most stable version for pygame)
2. Create Virtual Environment

### Prerequisites

```bash
pip install pygame numpy
```

### Installation

1. Clone or download the code
2. Run the program:

```bash
python maze_racing.py
```

## 🎯 How It Works

### Training Phase (20,000 steps)

The AI agents use **Q-Learning** to learn optimal maze navigation:

- **Exploration vs Exploitation**: Agents balance trying new paths (exploration) with using known good paths (exploitation)
- **Reward System**:
  - +200 for reaching the goal
  - +5 for moving closer to the goal
  - -1 for each step (encourages efficiency)
  - -8 for revisiting cells
  - -20 for hitting walls
- **Epsilon Decay**: Exploration rate decreases from 90% to 1% as training progresses

### Racing Phase

After training completes:
- All AIs use their learned Q-values to navigate
- Pure exploitation (no random exploration)
- Visual trails show each AI's path
- Leaderboard ranks by steps taken
- Efficiency calculated as: `(optimal_path_length / steps_taken) × 100%`

## 🎨 Interface

### Training View
- **Progress Bar**: Shows training completion percentage
- **AI Agent Panels**: Display each agent's:
  - Reward accumulation (positive rewards only)
  - Goals reached
  - Cells explored
  - Real-time position on maze

### Race View
- **Leaderboard**: Ranked by performance (🥇🥈🥉)
- **Efficiency Metrics**: How close each AI got to optimal
- **Race Statistics**:
  - Current frame / total frames
  - Progress percentage
  - Number of finishers
  - Optimal path length

## ⌨️ Controls

| Key | Action |
|-----|--------|
| **Mouse Wheel** | Scroll sidebar content |
| **↑ / ↓** | Scroll sidebar (alternative) |
| **← / →** | Adjust race speed (1x - 10x) |
| **R** | Restart race with same maze |
| **N** | Generate new maze and restart training |
| **Resize Window** | Drag window edges to resize |

## 🤖 AI Agents

All four agents use identical learning parameters to ensure fair competition:

| Agent | Color | Learning Rate | Discount Factor | Initial Exploration |
|-------|-------|---------------|-----------------|-------------------|
| **Explorer** | Red | 0.10 | 0.97 | 90% |
| **Sprinter** | Blue | 0.10 | 0.97 | 90% |
| **Balanced** | Green | 0.10 | 0.97 | 90% |
| **Adaptive** | Orange | 0.10 | 0.97 | 90% |

## 🧠 Technical Details

### Q-Learning Update Rule

```
Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
```

Where:
- **α (alpha)**: Learning rate (0.10)
- **γ (gamma)**: Discount factor (0.97)
- **r**: Reward received
- **s**: Current state (position)
- **a**: Action taken
- **s'**: Next state

### Maze Generation

- Uses **recursive backtracking** algorithm
- 21×21 grid with guaranteed path from start to goal
- Random loops added for complexity
- BFS calculates optimal path length

### Anti-Loop Mechanisms

During racing, AIs avoid getting stuck using:
- Heavy penalties for revisiting cells (100×)
- Deterministic action selection (always picks best Q-value)
- Visit counter to discourage loops

## 📊 Performance Metrics

- **Goals Reached**: Number of successful maze completions during training
- **Positive Rewards**: Sum of all beneficial rewards earned
- **Cells Explored**: Unique maze positions visited
- **Steps Taken**: Total moves to reach goal in race
- **Efficiency**: Percentage of optimal performance

## 🎨 Customization

You can easily modify these constants in the code:

```python
TRAIN_TIME = 20_000        # Training steps
MAX_RACE_STEPS = 800       # Maximum steps allowed in race
ROWS = COLS = 21           # Maze dimensions
RACE_SPEED = 1             # Initial race playback speed
TRAIL_LENGTH = 40          # Length of visual trail
```

## 🐛 Troubleshooting

**AIs get stuck in loops**: Increase `TRAIN_TIME` for more learning

**Race too fast/slow**: Use ← → arrow keys to adjust speed

**Scrolling not smooth**: Try updating pygame to latest version

**Window too small**: Resize the window or adjust `CELL` size constant

## 📝 Algorithm Notes

### Why Do All AIs Take The Same Path?

Since all agents have identical parameters and learn from the same maze, they converge to the same optimal policy. This demonstrates that Q-Learning can reliably find optimal solutions!

### Distance-Based Reward Shaping

The reward function includes a distance component:
```python
distance_reward = (old_distance - new_distance) × 5
```

This helps agents learn faster by providing continuous feedback about progress toward the goal.

## 🤝 Contributing

Feel free to fork and modify! Some ideas for enhancement:
- Different learning algorithms (SARSA, Deep Q-Learning)
- Larger/dynamic maze sizes
- Obstacles or moving hazards
- Multiple goals
- Agent vs Agent competition modes

## 📜 License

MIT License - Feel free to use this code for learning and experimentation!

## 🙏 Acknowledgments

Built with:
- **Pygame**: Graphics and game loop
- **NumPy**: Efficient array operations for Q-tables
- **Q-Learning**: Classic reinforcement learning algorithm

---

**Enjoy watching AI agents learn and compete!** 🎉

For questions or suggestions, feel free to open an issue or contribute improvements.