#!/bin/bash

# Script name: run_eetrack_play.sh

# Initialize variables
HEADLESS_FLAG=""
LOAD_RUN=""
NUM_ENVS=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --headless)
            HEADLESS_FLAG="--headless"
            shift
            ;;
        --load_run)
            if [[ -n "$2" ]]; then
                LOAD_RUN="--load_run $2"
                shift 2
            else
                echo "Error: --run_name requires an argument."
                exit 1
            fi
            ;;
        --num_envs)
            if [[ -n "$2" ]] && [[ "$2" =~ ^[0-9]+$ ]]; then
                NUM_ENVS="--num_envs $2"
                shift 2
            else
                echo "Error: --num_envs requires a valid integer."
                exit 1
            fi
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run the Python script with the appropriate flags
python3 source/standalone/workflows/rsl_rl/play.py --task eetrack-play --experiment_name g1_eetrack $HEADLESS_FLAG $LOAD_RUN $NUM_ENVS

