#!/bin/bash
#
# run_defense.sh is a script which executes the defense
#
# Envoronment which runs attacks and defences calls it in a following way:
#   run_defense.sh INPUT_DIR
# where:
#   INPUT_DIR - directory with input PNG images
#
# Note:
#   middle data will be saved in ./outputs
#   predict results will be saved in ./result.csv
#

INPUT_DIR=$1

rm -r ./outputs

python convert_to_pytorch_dataset.py \
    --input_dir="${INPUT_DIR}"

echo "Predicting via random smoothing ..."

python code/predict.py imagenet \
        ./models/imagenet/resnet50/noise_0.25/checkpoint.pth.tar 0.25 \
    ${INPUT_DIR} \
    --N 100 \
    --skip 100 \
    --batch 400 \

python defense_eval.py --input_dir="${INPUT_DIR}"