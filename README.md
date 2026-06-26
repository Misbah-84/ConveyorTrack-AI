# ConveyorTrack-AI 🚀

An industrial-grade computer vision system designed for real-time automated inventory counting and multi-class product classification on active conveyor belt lines. This project addresses the classic "flickering" and "+1/-1 counting jitter" common in traditional line-crossing tracking pipelines by employing robust spatial geometry and advanced tracking algorithms.

---

## 🛠️ Key Technical Features

* **Advanced Object Detection:** Fine-tuned utilizing the YOLO framework to accurately recognize overlapping, high-density consumer retail goods (beverages, pouches, packaged items).
* **Identity-Stability Tracking:** Configured with **BoT-SORT** integrating Camera Motion Compensation (CMC) to handle background vibrations caused by running conveyor machinery, preventing fragmented tracks and double-counting.
* **Spatial Polygon Verification:** Replaces fragile 1D "tripwires" with a 2D **Polygon Zone of Interest** using `cv2.pointPolygonTest` spatial calculus to verify product presence and trajectory over multiple sequential frames.
* **Class Locking & Consistency Logic:** Locks the definitive product classification after initial high-confidence identification frames to stop mid-stream class swapping.

---

## 📂 Project Structure

```text
├── config/
│   └── botsort.yaml          # Optimized tracking parameters for conveyor dynamics
├── data/
│   └── sample_video.mp4      # Test video of the conveyor belt line
├── weights/
│   └── best.pt               # Fine-tuned custom network weights
├── src/
│   ├── counting_pipeline.py  # Main execution script with spatial geometry logic
│   └── interactive_roi.py    # GUI utility to draw custom coordinate zones
├── requirements.txt          # Core dependencies
└── README.md                 # Documentation




📊 Performance Objectives: Actual vs. Predicted
Traditional vision models suffer from inventory mismatch due to bounding box jitter. This pipeline is engineered specifically to achieve zero-variance parity between the Ground Truth Physical Stock and Model Predicted Totals by enforcing multi-frame region confirmation.
