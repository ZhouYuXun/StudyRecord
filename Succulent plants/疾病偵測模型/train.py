from ultralytics import YOLO


# 訓練
model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='訓練的data.yaml', project="存訓練結果的路徑", epochs=150, imgsz=640)

# Predict with the model
model.export(format='torchscript')