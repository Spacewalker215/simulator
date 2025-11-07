# Autonomous Driving DonkeySim

## Installation

If you haven't installed the uv package manager yet, install it: `curl -LsSf https://astral.sh/uv/install.sh | sh`

Then clone the repository, cd into it, and install dependencies with `uv sync`

## Run Training

### Quick Start

1. Start the simulator on port 9091
2. Run training:

```bash
# Single environment (for testing)
uv run python src/algorithms/ppo_pufferlib.py --num-envs 1
```

> Parallel training is not yet working correctly.
```bash
# Parallel training (faster, recommended)
# Note: You need to start multiple simulator instances on ports 9091-9094
uv run python src/algorithms/ppo_pufferlib.py --num-envs 4
```

### Tensoboard

```bash
uv run tensorboard --logdir ./output/tensorboard/
```

Then open http://localhost:6006

### (deprecated) Stable Baselines3 Implementation

```bash
uv run ./src/algorithms/ppo_sb3.py --visualize
```