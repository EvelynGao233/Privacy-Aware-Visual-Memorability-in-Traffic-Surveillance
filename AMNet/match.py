import numpy as np
import pandas as pd
import os
import json
from glob import glob

# User paths (update these according to your dataset)
tracking_file = "c04x/c04x_mot.txt"
att_score_dir = "c04x/c04x_att_scores"
output_json = "c04x/object_memorability.json"
total_output_json = "c04x/total_memorability.json"

# Read tracking data (MOT file)
tracking_data = pd.read_csv(tracking_file, header=None)
tracking_data.columns = ["frame_id", "track_id", "x", "y", "w", "h", "confidence", "unused1", "unused2", "unused3"]

object_memorability = {}

att_files = sorted(glob(os.path.join(att_score_dir, "img*_step*.npy")))

for index, row in tracking_data.iterrows():
    frame_id = int(row["frame_id"]) 
    track_id = int(row["track_id"])
    x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
    frame_att_maps = sorted(glob(os.path.join(att_score_dir, f"img{frame_id:06d}_step*.npy")))

    if not frame_att_maps:
        continue

    # Load the last step attention map (most refined)
    att_maps = [np.load(att_map) for att_map in frame_att_maps]
    final_att_map = att_maps[-1]

    att_h, att_w = final_att_map.shape  

    # Scale
    # Original resolution of AIC21 videos
    original_img_w, original_img_h = 1280, 960
    scale_x = att_w / original_img_w 
    scale_y = att_h / original_img_h 

    # Calculate bounding box in attention map coordinates
    x1, y1 = int(x * scale_x), int(y * scale_y)
    x2, y2 = int((x + w) * scale_x), int((y + h) * scale_y)

    x1, x2 = max(0, x1), min(att_w, x2)
    y1, y2 = max(0, y1), min(att_h, y2)

    bbox_attention = final_att_map[y1:y2, x1:x2]

    # Compute average attention score for the object
    object_score = np.mean(bbox_attention) if bbox_attention.size > 0 else 0

    if track_id not in object_memorability:
        object_memorability[track_id] = []
    
    object_memorability[track_id].append({"frame": frame_id, "score": object_score})

with open(output_json, "w") as f:
    json.dump(object_memorability, f, indent=4)

print(f"Memorability scores saved to {output_json}")

total_memorability = {
    track_id: sum(entry["score"] for entry in frames)
    for track_id, frames in object_memorability.items()
}

with open(total_output_json, "w") as f:
    json.dump(total_memorability, f, indent=4)

print(f"Total memorability scores saved to {total_output_json}")
