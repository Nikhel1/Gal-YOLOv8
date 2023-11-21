# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#from colormap import random_color
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import glob

import matplotlib.pyplot as plt

import os
import os.path as osp
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import faster_coco_eval.core.mask as mask_util
#import faster_coco_eval
#from faster_coco_eval import COCO
#from faster_coco_eval.extra import PreviewResults
#from pycocotools.coco import COCO 
#from pycocotools.cocoeval import COCOeval

data_keyword = "test"

#Following numbers from conf mat plot in runs/detect/test3
confusion_matrix = np.array([[216, 1, 14, 16, 85, 38],
                            [9, 34, 7, 27, 53, 13],
                            [28, 13, 17, 19, 55, 15],
                            [14, 5, 3, 98, 98, 24]])

# Get the list of category IDs in the ground truth annotations
catIds = [2, 1, 3, 4] #coco_gt.getCatIds()
class_names = ['FR-I', 'FR-II', 'FR-X', 'R']
num_imgs_p_class = []
#for catId in range(1,len(catIds)+1):
#	num_imgs_p_class.append(len(coco_gt.getImgIds(catIds=catId)))
num_imgs_p_class = np.array([285, 90, 92, 144])

# Normalize the confusion matrix
confusion_matrix_norm = confusion_matrix/np.array(num_imgs_p_class)[:, np.newaxis]
print(confusion_matrix_norm.diagonal())
# Change names to class_names
FP = confusion_matrix_norm[:,4]; FP[[0, 1]] = FP[[1, 0]]
FN = confusion_matrix_norm[:,5]; FN[[0, 1]] = FN[[1, 0]]
class_names_json = ['FR-II', 'FR-I', 'FR-X', 'R']
mapping = [class_names_json.index(label) for label in class_names]
confusion_matrix_norm = confusion_matrix_norm[mapping][:, mapping]
confusion_matrix_norm = np.hstack((confusion_matrix_norm, np.array(FP).reshape(-1, 1), np.array(FN).reshape(-1, 1)))
print(confusion_matrix_norm)

class_names_new = ['FR-I', 'FR-II', 'FR-x', 'R', 'FP', 'FN']
#class_names_new = class_names 

# Following numbers from normalized plot in runs/detect/test3 
confusion_matrix_norm[0,4] = 0.18; confusion_matrix_norm[1,4] = 0.29
confusion_matrix_norm[2,4] = 0.19; confusion_matrix_norm[3,4] = 0.34

# Create a figure and axis
fig, ax = plt.subplots(figsize=(7, 7))

# Create a color-coded text representation of the confusion matrix [:,:-2]
im = ax.imshow(confusion_matrix_norm, cmap='Blues', vmin = 0, vmax = 1)

# Set axis labels and title
ax.set_xlabel('Predicted Class', fontsize=16)
ax.set_ylabel('True Class', fontsize=16)	
if data_keyword == 'test':
	data_keyword = 'Testing set'
if data_keyword == 'train':
	data_keyword = 'Training set'
ax.set_title(f'Gal-YOLOv8', fontsize=16)

# Set x and y axis tick labels and their font size
ax.set_xticks(np.arange(len(class_names_new)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names_new, fontsize=14)
ax.set_yticklabels(class_names, fontsize=14)

# Rotate x-axis tick labels for better readability
plt.xticks(rotation=45)

# Loop over data dimensions and create text annotations
for i in range(len(class_names)):
	for j in range(len(class_names_new)):
		text = ax.text(j, i, format(confusion_matrix_norm[i, j], '.2f'), ha='center', va='center', color='black')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
# Add a colorbar
cbar = fig.colorbar(im, cax=cax)
# Increase font size of colorbar tick labels
cbar.ax.tick_params(labelsize=14)

plt.savefig(f"Confusion_testing_set_yolo.pdf", bbox_inches='tight')

# Show the plot
plt.show()
