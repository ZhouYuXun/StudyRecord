from ultralytics import YOLO
import cv2

# 預測
# Load a model
model = YOLO('你訓練的/train/weights/best.pt路徑')  # load a custom model

results = model.predict("預測圖片路徑.jpg",
                        imgsz=640,
                        conf=0.5)

# 預測圖片存檔
res_plotted = results[0].plot()
cv2.imwrite('預測圖片存檔.jpg', res_plotted, [cv2.IMWRITE_JPEG_QUALITY, 80])