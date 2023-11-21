# Gal-YOLOv8
YOLOv8 implementation for galaxy detection

## Installation
Create a Python 3.10.9 environement with CUDA 11.6.2.
Then, install PyTorch 1.5.1+ and torchvision 0.6.1+:
```
conda install -c pytorch pytorch torchvision
```

Also install ultralytics:
```
pip install ultralytics
```

Install packages in requirements.txt.
```
pip install -r requirements.txt
```

## Data preparation

Download and extract RadioGalaxyNET data from [here](https://data.csiro.au/collection/61068).
We expect the directory structure to be the following:
```
./RadioGalaxyNET/
  annotations/  # annotation json files
  train/    # train images
  val/      # val images
  test/     # test images
```

### Use JSON2YOLO to change the annotations format to YOLO format and save it in 
YOLO_RadioGalaxyNET/datasets/labels/train

YOLO_RadioGalaxyNET/datasets/labels/val

YOLO_RadioGalaxyNET/datasets/labels/test

### Copy images from RadioGalaxyNET to 
YOLO_RadioGalaxyNET/datasets/images/train

YOLO_RadioGalaxyNET/datasets/images/val

YOLO_RadioGalaxyNET/datasets/images/test


## Training

To train on a single node with single gpu run:
```
python train.py
```

## Evaluation
To evaluate on test images with a single GPU run:
```
yolo task=detect mode=val model=runs/detect/train3/weights/best.pt data=RadioGalaxyNET.yaml
```

## License
MIT license.
