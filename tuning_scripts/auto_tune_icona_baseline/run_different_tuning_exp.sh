#!/bin/bash

# sbatch <runscript> <simulation duration> <initial parameter set>
sbatch /work/bd1179/b309170/icon-ml_models/icon-a-ml/run/exp.auto_tune_baseline.run day orig
sbatch /work/bd1179/b309170/icon-ml_models/icon-a-ml/run/exp.auto_tune_baseline.run week day_refined
sbatch /work/bd1179/b309170/icon-ml_models/icon-a-ml/run/exp.auto_tune_baseline.run month week_refined
sbatch /work/bd1179/b309170/icon-ml_models/icon-a-ml/run/exp.auto_tune_baseline.run year month_refined