#!/bin/bash

# Run multiple experiments with different hyperparameter combinations
# Usage: bash run_experiments.sh

echo "Starting hyperparameter sweep for auxiliary loss coefficients..."

# Array of load balance coefficients to test
load_balance_coefs=(0.000005 0.00001 0.000015 0.00003 0.00005 0.0001 0.001 0.01)

for load_balance_coef in "${load_balance_coefs[@]}"; do
    # Create a readable run_id from the parameters
    run_id="loadbal_${load_balance_coef}"
    
    echo ""
    echo "=========================================="
    echo "Running experiment: $run_id"
    echo "Load balance coef: $load_balance_coef"
    echo "=========================================="
    echo ""
    
    # Run the training script with current parameters
    python train_agents.py \
        --load_balance_coef "$load_balance_coef" \
        --run_id "$run_id"
    
    # Check if the training was successful
    if [ $? -eq 0 ]; then
        echo "Successfully completed: $run_id"
    else
        echo "Failed: $run_id"
        # Uncomment the next line if you want to stop on first failure
        # exit 1
    fi
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
