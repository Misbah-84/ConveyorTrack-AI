# ConveyorTrack-AI 🚀

An industrial-grade computer vision system for real-time automated inventory counting and multi-class product classification on active conveyor belt production lines. The system addresses common counting failures such as object flickering, identity fragmentation, and ±1 counting jitter by combining robust multi-object tracking with spatial geometry verification and temporal consistency logic.

---

# 🏗️ System Architecture

```text
               [ Industrial Conveyor Camera ]
                         │ (USB / RTSP)
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                 Computer Vision Processing Node              │
│                                                              │
│ • YOLO Fine-Tuned Object Detection                           │
│ • BoT-SORT Multi-Object Tracking                             │
│ • Camera Motion Compensation (CMC)                           │
│ • Polygon ROI Verification                                   │
│ • Product Class Locking Engine                               │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│             Inventory Counting & Analytics Engine            │
│                                                              │
│ • Multi-Frame Region Confirmation                            │
│ • Duplicate Count Prevention                                 │
│ • Stable Product Classification                              │
│ • Real-Time Inventory Statistics                             │
└──────────────────────────────────────────────────────────────┘
```

---

# 🛠️ Key Technical Features

## 1. Advanced Object Detection

A fine-tuned YOLO model optimized for industrial conveyor environments, accurately detecting overlapping and densely packed consumer products including:

- Beverage bottles
- Drink cans
- Pouches
- Packaged retail goods
- Mixed inventory items

The detector maintains high precision even under partial occlusion and varying illumination conditions.

---

## 2. Identity-Stable Multi-Object Tracking

BoT-SORT is configured with Camera Motion Compensation (CMC) to maintain persistent object identities despite conveyor vibration and camera movement.

Key capabilities include:

- Stable Track IDs
- Reduced ID switching
- Elimination of fragmented trajectories
- Prevention of duplicate inventory counts

---

## 3. Spatial Polygon Verification

Instead of relying on conventional 1D counting lines, the system uses a 2D Polygon Region of Interest implemented with:

```python
cv2.pointPolygonTest()
```

This spatial verification approach:

- Confirms object presence over multiple frames
- Prevents false trigger events
- Eliminates counting jitter
- Improves counting robustness in crowded scenes

---

## 4. Class Locking & Temporal Consistency

Once an object achieves a predefined confidence threshold across consecutive frames, its product category is permanently locked.

Benefits include:

- No mid-stream class swapping
- Stable inventory categorization
- Increased classification reliability
- Consistent reporting throughout each object's lifecycle

---

# 📂 Project Structure

```text
ConveyorTrack-AI/
├── config/
│   └── botsort.yaml
│
├── data/
│   └── sample_video.mp4
│
├── weights/
│   └── best.pt
│
├── src/
│   ├── counting_pipeline.py
│   └── interactive_roi.py
│
├── requirements.txt
└── README.md
```

---

# 📊 Performance Objectives

Traditional conveyor counting systems frequently experience inventory mismatches due to bounding box oscillation, unstable tracking identities, and repeated line-crossing events.

This pipeline is engineered to achieve near-zero variance between physical inventory counts and model predictions through:

- Multi-frame region confirmation
- Stable object identity preservation
- Polygon-based spatial validation
- Temporal class consistency
- Duplicate count suppression

The result is highly reliable, production-ready inventory monitoring suitable for automated manufacturing and warehouse environments.

---


---
