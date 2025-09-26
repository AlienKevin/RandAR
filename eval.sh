#!/bin/bash

# Enable error logging
set -e
set -o pipefail

# GCS Authentication - uncomment and set the path to your service account key
# export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"

# Arrays to track success and failure
successful_runs=()
failed_runs=()

# Function to log with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to create sample directory if it doesn't exist (only for local paths)
create_sample_dir() {
    local sample_dir="$1"
    # Only create directory if it's a local path (not GCS)
    if [[ ! "$sample_dir" =~ ^gs:// ]]; then
        if [ ! -d "$sample_dir" ]; then
            log_with_timestamp "Creating local sample directory: $sample_dir"
            mkdir -p "$sample_dir"
        fi
    else
        log_with_timestamp "Using GCS sample directory: $sample_dir"
    fi
}

log_with_timestamp "Starting evaluation sweep across num-inference-steps: 1 2 4 8 16 32 64 128 256"

# Sweep over different num-inference-steps values
for num_steps in 1 2 4 8 16 32 64 128 256; do
    log_with_timestamp "Starting evaluation with num-inference-steps: $num_steps"
    
    # Create sample directory
    folder="gs://randar"
    sample_dir="${folder}/num_inference_steps_${num_steps}"
    create_sample_dir "$sample_dir"
    
    # Run torchrun with error handling
    if torchrun --nproc-per-node=auto tools/search_cfg_weights.py \
        --config configs/randar/randar_l_0.3b_llamagen.yaml \
        --exp-name "randar_0.3b_360k_llamagen_${num_steps}" \
        --gpt-ckpt temp/randar_0.3b_llamagen_360k_bs_1024_lr_0.0004.safetensors \
        --vq-ckpt temp/vq_ds16_c2i.pt \
        --per-proc-batch-size 128 \
        --num-fid-samples-search 10000 \
        --num-fid-samples-report 50000 \
        --results-path results \
        --ref-path temp/VIRTUAL_imagenet256_labeled.npz \
        --sample-dir "$sample_dir" \
        --num-inference-steps $num_steps \
        --cfg-scales-interval 0.2 \
        --cfg-scales-search 3.0,5.0; then
        # --cfg-optimal-scale 3.4
        
        log_with_timestamp "✓ SUCCESS: Completed evaluation with num-inference-steps: $num_steps"
        successful_runs+=($num_steps)
    else
        log_with_timestamp "✗ FAILED: Evaluation failed with num-inference-steps: $num_steps (exit code: $?)"
        failed_runs+=($num_steps)
        log_with_timestamp "Continuing with next configuration..."
    fi
    
    echo "----------------------------------------"
done

# Print final summary
log_with_timestamp "EVALUATION SWEEP COMPLETED"
echo "========================================"

if [ ${#successful_runs[@]} -gt 0 ]; then
    log_with_timestamp "✓ SUCCESSFUL RUNS (${#successful_runs[@]}): ${successful_runs[*]}"
else
    log_with_timestamp "✗ NO SUCCESSFUL RUNS"
fi

if [ ${#failed_runs[@]} -gt 0 ]; then
    log_with_timestamp "✗ FAILED RUNS (${#failed_runs[@]}): ${failed_runs[*]}"
    echo "Check logs above for error details."
    exit 1
else
    log_with_timestamp "✓ ALL RUNS COMPLETED SUCCESSFULLY!"
fi
