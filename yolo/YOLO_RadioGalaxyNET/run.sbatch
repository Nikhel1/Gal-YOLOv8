#!/bin/bash -l
#SBATCH --account=OD-217715
#SBATCH --job-name=yolo8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80gb
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nikhel.gupta@csiro.au

echo 'Starting'

source /home/gup039/.bashrc

# Delete the settings file to avoid conflicts with the data folder
rm ~/.config/Ultralytics/settings.yaml

# Train, save and validate the model on val and test sets
python train.py

# Evaluation (not required, can see in slurmxxx.txt file)
# yolo task=detect mode=val model=runs/detect/train3/weights/best.pt data=RadioGalaxyNET.yaml
# yolo task=detect mode=val model=runs/detect/train3/weights/best.pt data=RadioGalaxyNET.yaml split=test save_json=True --conf=0.06
