#!/bin/bash

# Run multiple experiments with different hyperparameter combinations
# Usage: bash run_experiments.sh


# Array of load balance coefficients to test
atari_games=("Asteroids" "BeamRider" "Bowling" "Boxing" "DemonAttack" "FishingDerby" "Freeway" "MontezumaRevenge" "Qbert" "Seaquest" "SpaceInvaders")

for game in "${atari_games[@]}"; do
    echo "Running experiment for $game..."
    python train_agents.py --env "${game}NoFrameskip-v4" --mode wsa --run_id "wsa-${game}"
    python train_agents.py --env "${game}NoFrameskip-v4" --mode ppo --run_id "ppo-${game}"   
done

