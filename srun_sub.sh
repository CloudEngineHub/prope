#!/bin/bash

NUM_GPUS=2

CONTAINER_IMAGE=/lustre/fsw/portfolios/nvr/users/qiwu/containers/nre-ord-dev-v0.sqsh
SRUN_TIMEOUT=14400  # 4 hours in seconds

PRE_COMMAND=(
    "source /lustre/fsw/portfolios/nvr/users/ruilongl/miniforge3/etc/profile.d/conda.sh &&"
    "conda activate prope &&"
    "echo 'conda activated'"
)
COMMAND=(
    # "bash ./scripts/nvs.sh --ray_encoding plucker --pos_enc prope --gpus '0,1'" 
    # "bash ./scripts/nvs.sh --ray_encoding plucker --pos_enc gta --gpus '0,1'"

    # "bash ./scripts/nvs.sh --ray_encoding plucker --pos_enc none --gpus '0,1' --test-zoom-in '1 3 5' &&"
    # "bash ./scripts/nvs.sh --ray_encoding plucker --pos_enc none --gpus '0,1' --test-context-views '2 4 8 16'"

    # "bash ./scripts/nvs.sh --ray_encoding camray --pos_enc prope --gpus '0,1' --test-zoom-in '1 3 5' &&"
    # "bash ./scripts/nvs.sh --ray_encoding camray --pos_enc prope --gpus '0,1' --test-context-views '2 4 8 16'"

    # "bash ./scripts/nvs.sh --ray_encoding camray --pos_enc gta --gpus '0,1' --test-zoom-in '1 3 5' &&"
    # "bash ./scripts/nvs.sh --ray_encoding camray --pos_enc gta --gpus '0,1' --test-context-views '2 4 8 16'"

    # "bash ./scripts/nvs.sh --ray_encoding camray --pos_enc none --gpus '0,1' --test-zoom-in '1 3 5' &&"
    # "bash ./scripts/nvs.sh --ray_encoding camray --pos_enc none --gpus '0,1' --test-context-views '2 4 8 16'"

    # "bash ./scripts/nvs.sh --ray_encoding raymap --pos_enc prope --gpus '0,1' --test-zoom-in '1 3 5' &&"
    # "bash ./scripts/nvs.sh --ray_encoding raymap --pos_enc prope --gpus '0,1' --test-context-views '2 4 8 16'"

    # "bash ./scripts/nvs.sh --ray_encoding raymap --pos_enc gta --gpus '0,1' --test-zoom-in '1 3 5' &&"
    # "bash ./scripts/nvs.sh --ray_encoding raymap --pos_enc gta --gpus '0,1' --test-context-views '2 4 8 16'"

    # "bash ./scripts/nvs.sh --ray_encoding raymap --pos_enc none --gpus '0,1' --test-zoom-in '1 3 5' &&"
    # "bash ./scripts/nvs.sh --ray_encoding raymap --pos_enc none --gpus '0,1' --test-context-views '2 4 8 16'"

    # "bash ./scripts/nvs.sh --ray_encoding none --pos_enc prope --gpus '0,1' --test-zoom-in '1 3 5' &&"
    # "bash ./scripts/nvs.sh --ray_encoding none --pos_enc prope --gpus '0,1' --test-context-views '2 4 8 16'"

    # "bash ./scripts/nvs.sh --ray_encoding none --pos_enc gta --gpus '0,1' --test-zoom-in '1 3 5' &&"
    # "bash ./scripts/nvs.sh --ray_encoding none --pos_enc gta --gpus '0,1' --test-context-views '2 4 8 16'"
)
AFTER_COMMAND=(
    "echo 'job done'"
)
COMMAND_STRING="${PRE_COMMAND[*]} && ${COMMAND[*]} && ${AFTER_COMMAND[*]}"

echo "Running command: "
echo "--------------------------------"
echo "$COMMAND_STRING"
echo "--------------------------------"



# Create a temporary job script
cat << EOF > job_script.sh
#!/bin/bash
#SBATCH --account=nvr_torontoai_videogen
#SBATCH --gpus-per-node=$NUM_GPUS
#SBATCH --partition=polar,grizzly
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm-%j.out    # %j will be replaced with the job ID
#SBATCH --error=logs/slurm-%j.err     # Separate file for errors

timeout $SRUN_TIMEOUT srun \
--container-image=$CONTAINER_IMAGE \
--container-mounts=$HOME:/root,/lustre:/lustre \
--container-workdir=/lustre/fsw/portfolios/nvr/users/ruilongl/prope \
bash -c "$COMMAND_STRING"

# Check if timeout occurred
if [ \$? -eq 124 ]; then
    echo "Command timed out after $SRUN_TIMEOUT seconds, Requeueing job \$SLURM_JOB_ID"
    scontrol requeue \$SLURM_JOB_ID
fi
EOF

# Submit the job
sbatch job_script.sh