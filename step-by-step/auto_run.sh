#!/bin/bash
#SBATCH --time 02:00:00 # time, specify max time allocation
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=sara.althubaiti@kaust.edu.sa ##specify your e-mail address 
#SBATCH --job-name=deepMOCCA
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=70GB
#SBATCH --array=0-32 ## 1 is the last index in the input file (e.g. OA.files.txt)
echo "Job ID=$SLURM_JOB_ID,  Running task:$SLURM_ARRAY_TASK_ID"

values=$(grep "^${SLURM_ARRAY_TASK_ID}:" /ibex/scratch/projects/c2014/sara/can_list.txt)
echo $values
filename=$(echo $values | cut -f 2 -d:)
echo $filename
source activate /home/althubsw/miniconda3/envs/test
args=("$@")
python -u /ibex/scratch/projects/c2014/sara/deepmocca_training_one_feature_new_idea_more_layers.py $filename ${args[0]} ${args[1]} ${args[2]} > /ibex/scratch/projects/c2014/sara/new_idea_one_feature_more_layers/"$filename"_${args[3]}.txt
