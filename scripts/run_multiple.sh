#!/bin/bash

# name of the experiment (should match a config file in configs/)
DATASET=$1
EXP=$2

BATCH_TEMPLATE="scripts/run_multiple_slurm.batch" # path to the batch template.
BATCH_SIZE=30 # for ILIAS, dont go too lower than 30 as it slows down due to loading
RESULTS_FOLDER=results/ # change it based on where you want to save the results

# Loop through the list pairwise
for START in `seq 0 $BATCH_SIZE 1250`
do
  END=$((START + $BATCH_SIZE))

  # submit the job
  sbatch $BATCH_TEMPLATE $DATASET $EXP $START $END $RESULTS_FOLDER 
done
