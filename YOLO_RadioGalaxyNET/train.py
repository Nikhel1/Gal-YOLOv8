from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="RadioGalaxyNET.yaml", epochs=50)  # train the model

print ('Validation set metrics ------------------------------------------------:')
metrics = model.val()

print ('Test set metrics ------------------------------------------------:')
metrics = model.val(split='test', save_json=True, conf=0.06)
