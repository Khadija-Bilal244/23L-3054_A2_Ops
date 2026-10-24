#!/usr/bin/env bash
# >= 6 runs, each differing from a reference by ONE meaningful change.
set -e
T="python -m src.train"
$T --model baseline  --run-name e1_baseline_cnn
$T --model transfer  --run-name e2_mnv2_ref                       # reference
$T --model transfer  --run-name e3_mnv2_lr3e-4  --lr 3e-4         # learning rate
$T --model transfer  --run-name e4_mnv2_noaug   --no-aug          # augmentation
$T --model transfer  --run-name e5_mnv2_unfreeze40 --unfreeze-n 40  # frozen layers
$T --model transfer  --run-name e6_mnv2_size160 --size 160        # image size
$T --model transfer  --backbone efficientnetb0 --run-name e7_effb0 # backbone
$T --model transfer  --run-name e8_mnv2_autolog --tracking autolog # Task 3.3 second approach
