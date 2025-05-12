#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 --run_name [RUN_NAME] [other arguments]"
    exit 1
}

# Check if --run_name argument is provided
RUN_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_name)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                RUN_NAME="$2"
                shift 2
            else
                echo "Error: --run_name requires a non-empty argument."
                usage
            fi
            ;;
        *)
            OTHER_ARGS+="$1 "
            shift
            ;;
    esac
done

# If RUN_NAME is still empty, show usage and exit
if [[ -z "$RUN_NAME" ]]; then
    echo "Error: --run_name is a mandatory argument."
    usage
fi

# Execute the command with --run_name
python3 source/standalone/workflows/rsl_rl/train.py \
    --task eetrack-train \
    --logger wandb \
    --log_project_name eetrack \
    --headless \
    --max_iterations 50000 \
    --run_name "$RUN_NAME" $OTHER_ARGS

