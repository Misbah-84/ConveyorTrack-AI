from ultralytics import YOLO
import cv2

# 1. PATHS
MODEL_PATH = r'D:\Projects\Yolo\last.pt' 
VIDEO_PATH = r'D:\Projects\Yolo\IMG_7262.MOV'

# 2. LOAD MODEL
model = YOLO(MODEL_PATH)

# 3. RUN WITH TRACKER
# persist=True tells the model to remember IDs across frames
results = model.track(
    source=VIDEO_PATH,
    conf=0.3,             # Slightly higher confidence for tracking stability
    iou=0.5,              # Intersection over Union
    show=True,            # Watch the ID numbers live
    save=True, 
    tracker="botsort.yaml", # You can also use "bytetrack.yaml"
    project=r'D:\Projects\Yolo\Tracking_Results',
    name='Conveyor_ID_Test'
)

print("✅ Tracking complete. Check the 'Tracking_Results' folder.")