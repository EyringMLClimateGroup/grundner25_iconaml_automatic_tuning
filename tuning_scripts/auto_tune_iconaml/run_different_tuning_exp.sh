#!/bin/bash

# sbatch <job> <simulation duration> <initial parameter set>
sbatch /work/bd1179/b309170/icon-ml_models/icon-a-ml/run/exp.auto_tune.run day orig # Output in log.auto_tune.12876603.o
sbatch /work/bd1179/b309170/icon-ml_models/icon-a-ml/run/exp.auto_tune.run week refined_day # Output in log.auto_tune.12889494.o
sbatch /work/bd1179/b309170/icon-ml_models/icon-a-ml/run/exp.auto_tune.run month refined_week # Output in log.auto_tune.12892616.o
sbatch /work/bd1179/b309170/icon-ml_models/icon-a-ml/run/exp.auto_tune.run year refined_month # Output in log.auto_tune.12956787.o
sbatch /work/bd1179/b309170/icon-ml_models/icon-a-ml/run/exp.auto_tune.run year refined_month_240925 # Output in log.auto_tune.12962670.o