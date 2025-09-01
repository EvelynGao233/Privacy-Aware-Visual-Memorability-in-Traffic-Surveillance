# Privacy-Aware Visual Memorability in Traffic Surveillance

This repository provides the code and scripts used in our project.  
It integrates multiple open-source frameworks for multi-camera vehicle tracking, visual memorability estimation, and vehicle attribute analysis.  

---

## 🔹 Requirements
Please follow the installation instructions in each linked repository below.  

---

## 🔹 Pipeline Overview

### 1. Tracking with [AIC21-MTMC](https://github.com/LCFractal/AIC21-MTMC)
- Run MTMC on the AIC21 dataset (we mainly use test videos `c041–c046`).  
- The output is MOT-format tracking files (e.g., `c041_mot.txt`), which serve as the basis for later memorability computation.

---

### 2. Memorability Prediction with [AMNet](https://github.com/ok1zjf/AMNet)
- Run AMNet on frames extracted from MTMC videos.  
- For each frame, AMNet generates:
  - A scalar memorability score (not used here).
  - Spatial attention maps.

> **Modification**: we updated `amnet.py` → `PredictionResult.get_attention_maps()`  
to support saving raw attention maps as `.npy` files (`--att-scores-out`).  
These `.npy` maps are later aligned with MOT bounding boxes.

---

### 3. Object-Level Memorability (`match.py`)
- `match.py` matches MOT bounding boxes with AMNet attention maps.  
- For each object in each frame, we compute the **average attention value** inside its bounding box.  
- Outputs:
  - `object_memorability.json` – frame-level memorability scores per object.
  - `total_memorability.json` – aggregated memorability score per object.

---

### 4. Vehicle Attributes
We enrich the tracked vehicles with additional visual properties:
- [Vehicle-Make-Color-Recognition](https://github.com/nikalosa/Vehicle-Make-Color-Recognition) → predicts vehicle color.  
- [fast-neural-style-pytorch](https://github.com/rrmina/fast-neural-style-pytorch) → applies style transfer to augment frames.

---

### 5. Analysis
Scripts in `/analysis/` combine memorability results with vehicle attributes to provide:  
- Distribution statistics.  
- Comparisons across vehicle colors and styles.  

---

## Related Repositories
- [AIC21-MTMC](https://github.com/LCFractal/AIC21-MTMC)
- [AMNet](https://github.com/ok1zjf/AMNet)
- [Vehicle-Make-Color-Recognition](https://github.com/nikalosa/Vehicle-Make-Color-Recognition)
- [fast-neural-style-pytorch](https://github.com/rrmina/fast-neural-style-pytorch)

