#!/bin/bash

# Run multiple experiments with different hyperparameter combinations
# Usage: bash run_experiments.sh

echo "Starting hyperparameter sweep for auxiliary loss coefficients..."

# Array of entropy coefficients to test
entropy_coefs=(0.0 0.00001 0.00005 0.0001 0.0005 0.001)

# Array of load balance coefficients to test
load_balance_coefs=(0.0 0.000005 0.00001 0.000015 0.00003 0.00005)

# Run experiments with different combinations
for entropy_coef in "${entropy_coefs[@]}"; do
    for load_balance_coef in "${load_balance_coefs[@]}"; do
        # Create a readable run_id from the parameters
        run_id="entropy_${entropy_coef}_loadbal_${load_balance_coef}"
        
        echo ""
        echo "=========================================="
        echo "Running experiment: $run_id"
        echo "Entropy coef: $entropy_coef"
        echo "Load balance coef: $load_balance_coef"
        echo "=========================================="
        echo ""
        
        # Run the training script with current parameters
        python train_agents.py \
            --entropy_coef "$entropy_coef" \
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
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
