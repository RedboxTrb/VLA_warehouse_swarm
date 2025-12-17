# Warehouse Drone Swarm - Simulation Project

## Overview
Autonomous multi-agent drone system for GPS-denied indoor warehouse navigation using:
- Reinforcement Learning (RL) for low-level control
- Vision-Language-Action (VLA) for high-level planning
- Multi-agent coordination (MAPPO)
- Realistic sensor fusion (EKF + learned)

## Environment
- Ubuntu 22.04
- CUDA 12.1+
- Python 3.10+
- Conda environment: torch
- GPU: NVIDIA RTX 5000 Ada

## Project Structure
```
warehouse_swarm/
├── envs/           # Custom Gymnasium environments
├── models/         # Neural network architectures
├── training/       # Training scripts
├── utils/          # Helper functions and tools
├── configs/        # Configuration files
├── experiments/    # Training logs and checkpoints
│   ├── logs/
│   ├── checkpoints/
│   └── videos/
├── data/           # Datasets and demonstrations
│   ├── demonstrations/
│   └── sensor_logs/
└── docs/           # Documentation
```


## Getting Started
```bash
# Activate conda environment
conda activate torch

# Navigate to project
cd ~/warehouse_swarm
```

## Author
Kushal - MNIT
Location: Khobar, Eastern Province, SA

## Research Goal
Publication at ICRA/IROS/CoRL - Budget-constrained swarm system with natural language control
