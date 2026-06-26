import cv2
import os
import csv
import glob
from ultralytics import YOLO, solutions
from collections import Counter
from datetime import datetime

# 1. PATH CONFIGURATION
base_dir = r"D:\Projects\Yolo"
search_dir = r"D:\Projects\Yolo\testing\Actuall" 
output_videos_dir = os.path.join(base_dir, "Audit_Videos")
output_csv_dir = os.path.join(base_dir, "Inventory_Reports")

os.makedirs(output_videos_dir, exist_ok=True)
os.makedirs(output_csv_dir, exist_ok=True)

model = YOLO(os.path.join(base_dir, "best.pt"))
csv_path = os.path.join(output_csv_dir, f"Consensus_Audit_{datetime.now().strftime('%H%M%S')}.csv")

# Initialize CSV with Headers
with open(csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Subfolder", "Video_Source", "Product_Name", "Verified_Count"])

# Find all videos recursively
video_files = []
for ext in ('*.mp4', '*.MP4', '*.mov', '*.MOV'):
    video_files.extend(glob.glob(os.path.join(search_dir, "**", ext), recursive=True))

print(f"🚀 Found {len(video_files)} videos. Engaging Consensus Logic for Rotating Products...")

for video_path in video_files:
    video_name = os.path.basename(video_path)
    subfolder_name = os.path.basename(os.path.dirname(video_path))
    video_save_path = os.path.join(output_videos_dir, f"Audit_{video_name}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): continue
    w, h, fps = (int(cap.get(3)), int(cap.get(4)), int(cap.get(5)))

    # ROI: Extended vertically for maximum "Observation Time"
    region_points = [(int(w*0.05), int(h*0.1)), (int(w*0.95), int(h*0.1)), 
                     (int(w*0.95), int(h*0.98)), (int(w*0.05), int(h*0.98))]

    counter = solutions.ObjectCounter(
        region=region_points, model=os.path.join(base_dir, "best.pt"),
        show=False, tracker="botsort.yaml", conf=0.15
    )

    # --- STATEFUL MEMORY ---
    id_voting_box = {} # {track_id: [list of labels]}
    video_writer = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        results = counter.process(frame)
        annotated_frame = results.plot_im

        # Capture every single frame's prediction for every ID
        if hasattr(results, 'track_data') and results.track_data.track_ids is not None:
            for track_id, cls_idx in zip(results.track_data.track_ids, results.track_data.clss):
                class_name = model.names[int(cls_idx)]
                if track_id not in id_voting_box:
                    id_voting_box[track_id] = []
                id_voting_box[track_id].append(class_name)

        video_writer.write(annotated_frame)
        cv2.imshow("High-Vote Audit Engine", cv2.resize(annotated_frame, (1280, 720)))
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    # --- POST-PROCESS: CALCULATE TRUTH ---
    final_video_counts = {}
    
    # We look at every ID that was seen for a significant amount of time
    for track_id, labels in id_voting_box.items():
        if len(labels) > 20: # Must be seen for ~1 second to be a real product
            # MAJORITY VOTE: This solves the rotation flicker
            verified_class = Counter(labels).most_common(1)[0][0]
            final_video_counts[verified_class] = final_video_counts.get(verified_class, 0) + 1

    # Save to CSV immediately after video ends
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        for product, count in sorted(final_video_counts.items()):
            writer.writerow([timestamp, subfolder_name, video_name, product, count])
            print(f"   ✅ Logged {product}: {count}")

    cap.release()
    video_writer.release()

cv2.destroyAllWindows()
print(f"🏁 ALL VIDEOS PROCESSED. Report: {csv_path}")