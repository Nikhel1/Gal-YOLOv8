from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

# path to annotations file and result file
data_keyword = "test"
ann_file = "/home/gup039/EMU/EMUclassifier/GalDINO/RadioGalaxyNET_V4/annotations/instances_{}2017.json".format(data_keyword)

# load annotations and create COCO object
coco_gt = COCO(str(ann_file))

mp = {t['file_name']:t['id'] for t in coco_gt.dataset['images']}

coco_p = json.load(open("runs/detect/val3/predictions.json"))
for idx in range(len(coco_p)):
	coco_p[idx]['image_id'] = mp[coco_p[idx]['image_id']+'.png']
	coco_p[idx]['category_id'] = coco_p[idx]['category_id']+1

with open('runs/detect/val3/predictions_new.json', 'w') as fp:
	json.dump(coco_p, fp)

coco_dt = coco_gt.loadRes(str("runs/detect/val3/predictions_new.json"))

# load results and create COCO object

cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')
#cocoEval.params.catIds = [1]
cocoEval.params.imgIds = coco_gt.getImgIds()
cocoEval.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 24 ** 2], [24 ** 2, 48 ** 2], [48 ** 2, 1e5 ** 2]]
cocoEval.params.areaRngLbl = ['all', 'small', 'medium', 'large']
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
