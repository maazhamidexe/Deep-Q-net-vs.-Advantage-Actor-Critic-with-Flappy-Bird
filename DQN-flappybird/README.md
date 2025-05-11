# Deep Q-Network (DQN) Implementation

A PyTorch implementation of Deep Q-Learning with support for Double DQN and Dueling DQN architectures. This implementation is designed to work with Gymnasium environments, including Flappy Bird.

## Features

- Deep Q-Learning implementation with PyTorch
- Support for Double DQN to reduce overestimation of Q-values
- Support for Dueling DQN architecture
- Experience Replay Memory for stable training
- Configurable hyperparameters via YAML
- Automatic model saving and visualization of training progress
- Support for both training and evaluation modes
- GPU support (with CPU fallback)

## Requirements

- Python 3.x
- PyTorch
- Gymnasium
- NumPy
- Matplotlib
- PyYAML
- Flappy Bird Gymnasium (for Flappy Bird environment)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd dqn_pytorch
```

2. Install the required packages:
```bash
pip install torch gymnasium numpy matplotlib pyyaml flappy-bird-gymnasium
```

## Project Structure

```
dqn_pytorch/
├── agent.py              # Main agent implementation
├── dqn.py               # DQN neural network architecture
├── experience_replay.py # Experience replay memory implementation
├── hyperparameters.yml  # Configuration file for hyperparameters
└── runs/               # Directory for saved models and training logs
```

## Usage

### Training

To train the agent with a specific hyperparameter set:

```bash
python agent.py <hyperparameter_set> --train
```

### Evaluation

To evaluate a trained model:

```bash
python agent.py <hyperparameter_set>
```

### Hyperparameters

The hyperparameters are configured in `hyperparameters.yml`. You can create different sets of hyperparameters for different experiments. The configuration includes:

- Environment settings
- Learning rate
- Discount factor
- Network architecture parameters
- Training parameters (epsilon decay, batch size, etc.)
- DQN variants (Double DQN, Dueling DQN)

## Training Progress

During training, the following information is saved in the `runs` directory:

- Model checkpoints (`.pt` files)
- Training logs (`.log` files)
- Training progress graphs (`.png` files)

The graphs show:
- Mean rewards over episodes
- Epsilon decay over time

## Implementation Details

### DQN Architecture

The implementation includes:
- Policy network and target network
- Experience replay memory for stable training
- Epsilon-greedy exploration strategy
- Optional Double DQN and Dueling DQN architectures

### Training Process

1. The agent interacts with the environment using an epsilon-greedy policy
2. Experiences are stored in the replay memory
3. The policy network is trained using mini-batches from the replay memory
4. The target network is periodically synchronized with the policy network
5. Training progress is automatically saved and visualized

## License

[Add your license information here]

## Contributing

[Add contribution guidelines if applicable]

## Acknowledgments

- Based on the original DQN paper: "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)
- Uses the Gymnasium framework for environment simulation
