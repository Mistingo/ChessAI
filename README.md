# Mini AlphaZero Chess

AlphaZero-inspired chess engine implemented in Python using:

- Self-play reinforcement learning
- Monte Carlo Tree Search (MCTS)
- Policy + Value neural networks
- Dynamic exploration temperature
- Learning rate scheduling
- Training monitoring and evaluation tools
- Human vs AI gameplay

The objective of this project is to build an autonomous chess agent capable of learning strategies through repeated self-play matches and progressively improving its decision making.

---

## Features

### Reinforcement Learning

- Self-play training pipeline
- Reward shaping
- Autonomous learning
- Iterative model updates
- Checkpoint generation

### Search System

- Monte Carlo Tree Search (MCTS)
- Configurable simulation count
- Position exploration
- Search-guided move selection

### Neural Architecture

Policy network:

- predicts move probabilities

Value network:

- evaluates board positions

Combined architecture:

```text
Board state
      ↓
Neural Network
      ↓
Policy + Value outputs
      ↓
Monte Carlo Tree Search
      ↓
Move selection
```

### Evaluation Tools

- Human vs AI gameplay
- Playable interface
- Model testing
- Training statistics
- Learning visualisation
- Performance tracking

---

## Project Structure

```text
MiniAlphaZeroChess/
│
├── Play_agent.py
├── Train_selfplay.py
│
├── Models/
│   ├── alphazero_optimized_agent_final.pth
│   └── best_alphazero_optimized_agent.pth
│
├── Plots/
│   └── training_progress.png
│   
├── Results/
│   └── training_history.json
│
├── Assets/
│
└── README.md
```

---

## Training Workflow

The agent learns entirely through self-play.

```text
Initial model
      ↓
Self-play matches
      ↓
MCTS exploration
      ↓
Move selection
      ↓
Reward computation
      ↓
Policy update
      ↓
Evaluation
      ↓
Checkpoint saving
      ↓
Repeat
```

Each iteration improves the policy and value estimations used during search.

---

## Playing Against the Agent

After training, the model can be evaluated directly through an interactive interface.

File:

```text
Play_agent.py
```

Capabilities:

- Human vs AI games
- Loading trained checkpoints
- Observation of learned behaviour
- Real-time move selection
- Evaluation of training quality

Workflow:

```text
Training
      ↓
Checkpoint generation
      ↓
Model loading
      ↓
Playable interface
      ↓
Human vs Agent match
```

---

## Usage

### Train the model

```bash
python Train_selfplay.py
```

### Play against the trained model

```bash
python Play_agent.py
```

Training cycle:

```text
Self-play
      ↓
Learning
      ↓
Checkpoint generation
      ↓
Human evaluation
```

---

## Training Metrics

The project records multiple metrics during learning.

### Loss Tracking

- Total loss
- Policy loss
- Value loss

Observed example:

Initial total loss:

```text
≈ 8.31
```

Final total loss:

```text
≈ 2.76
```

Policy accuracy reached values close to:

```text
1.0
```

during late iterations.

---

## Evaluation Metrics

Collected metrics include:

### Performance

- Win rate
- Draw rate
- Loss rate
- Average game length

### Learning Quality

- Policy accuracy
- Entropy evolution
- Loss convergence
- Reward behaviour

### Training Control

- Learning rate scheduling
- Exploration temperature decay
- MCTS simulation tracking

Example values:

MCTS simulations:

```text
24 simulations
```

Temperature schedule:

```text
1.5 → 0.1
```

Training iterations:

```text
100+
```

---

## Training Visualisation

Training monitoring includes:

- Loss evolution
- Evaluation results
- Policy accuracy
- Entropy evolution
- Learning rate schedule
- Temperature decay
- Average game length
- MCTS statistics

Example:

<img src="https://github.com/ThomasAgd/Mini-alphazero-chess/blob/main/Plots/training_progress.png" alt="Training Progress" width="600" />

Visual outputs currently monitor:

- Total Loss
- Policy Loss
- Value Loss
- Evaluation Results
- Accuracy
- Entropy
- Learning Rate
- Temperature
- MCTS simulations
- Game lengths

---

## Experimental Components

Implemented:

✓ Self-play reinforcement learning

✓ Monte Carlo Tree Search

✓ Policy network

✓ Value network

✓ Temperature scheduling

✓ Learning rate decay

✓ Reward shaping

✓ Human evaluation interface

✓ Training monitoring

✓ Checkpoint testing

---

## Technologies

Language:

- Python

AI Methods:

- Reinforcement Learning
- Monte Carlo Tree Search
- Self-play learning
- Policy optimisation
- Value estimation

Domains:

- Artificial Intelligence
- Game AI
- Search algorithms
- Strategic optimisation
- Experimental machine learning

---

## Results Example

Observed behaviour during training:

- Stable loss convergence
- Reduction of exploration entropy
- Controlled temperature decay
- Consistent MCTS behaviour
- Increasing policy accuracy
- Progressive strategy stabilisation

---

## Future Improvements

Planned work:

- Stronger evaluation opponents
- ELO estimation
- Model comparison system
- Replay export
- Tournament mode
- Checkpoint competitions
- TensorBoard integration
- GPU optimisation
- Extended search depth
- Training dashboards

---

## Academic Context

Personal project exploring advanced AI approaches inspired by AlphaZero.

Main topics explored:

- Reinforcement learning
- Autonomous agents
- Chess AI
- Monte Carlo Tree Search
- Policy + Value architectures
- Self-play optimisation
- Experimental evaluation

---

## Author

Thomas Augendre
