#!/bin/bash
source envn/bin/activate
sbatch --mem=16G -p gpu-troja -G 2 jobs.sh
