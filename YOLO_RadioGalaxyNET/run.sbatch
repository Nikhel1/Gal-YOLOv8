#!/bin/bash -l
#SBATCH --job-name=yolo8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80gb
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

echo 'Starting'

# Delete the settings file to avoid conflicts with the data folder
rm ~/.config/Ultralytics/settings.yaml

# Train, save and validate the model on val and test sets
python train.py

# Evaluation (not required, can see in slurmxxx.txt file)
yolo task=detect mode=val model=runs/detect/train3/weights/best.pt data=RadioGalaxyNET.yaml
