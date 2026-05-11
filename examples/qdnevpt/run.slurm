#!/bin/bash
#SBATCH --time=0:05:00
#SBATCH --account=PAS1583
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=eigen_analysis_test
#SBATCH --mem=20G
#SBATCH --output=./slurm_out/%x_%j.out

start_time=$(date +%s)
echo ----
echo Job started at `date`
echo ----
echo This job is working on compute node `srun hostname |sort -n`

module reset
module load miniconda3/24.1.2-py310
conda activate prismdmet

export JOBNAME="test"

cd "$SLURM_SUBMIT_DIR"

export TMPDIR="/fs/scratch/PAS1583/"

echo "----"
echo "Running  script: $JOBNAME"
python "$JOBNAME.py" > "$JOBNAME.out"

echo "----"
echo "Job ended at $(date)"

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

h=$(( elapsed / 3600 ))
m=$(( (elapsed % 3600) / 60 ))
s=$(( elapsed % 60 ))

printf "Total Walltime: %02d:%02d:%02d\n" $h $m $s
echo "----"