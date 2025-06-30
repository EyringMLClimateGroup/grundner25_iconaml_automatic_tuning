#!/bin/bash

# sbatch <runscript> <simulation duration> <initial parameter set>
sbatch /work/bd1179/b309170/icon-ml_models/icon-a-ml/run/exp.auto_tune_baseline.run day orig # Output in log.auto_tune_baseline.13758205.o
sbatch /work/bd1179/b309170/icon-ml_models/icon-a-ml/run/exp.auto_tune_baseline.run week day_refined # Output in log.auto_tune_baseline.13774617.o
sbatch /work/bd1179/b309170/icon-ml_models/icon-a-ml/run/exp.auto_tune_baseline.run month week_refined # Output in log.auto_tune_baseline.13850767.o
sbatch /work/bd1179/b309170/icon-ml_models/icon-a-ml/run/exp.auto_tune_baseline.run year month_refined # Output in log.auto_tune_baseline.13852199.o
sbatch /work/bd1179/b309170/icon-ml_models/icon-a-ml/run/exp.auto_tune_baseline.run year year_refined # Output in log.auto_tune_baseline.17820065.o
sbatch /work/bd1179/b309170/icon-ml_models/icon-a-ml/run/exp.auto_tune_baseline.run year year_refined_toa # With crt/crs tweak. Output in log.auto_tune_baseline.13887792.o
