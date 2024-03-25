import json
import numpy as np
import cv2
from shapely.geometry import Polygon
import os

def json_to_mask(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    masks = {}
    for shape in data['shapes']:
        label = shape['label']
        group_id = shape.get('group_id', None)  # Get the group_id, default to None if not present
        polygon = shape['points']
        poly_np = np.array(polygon, dtype=np.int32)

        if group_id is not None:
            if group_id not in masks:
                masks[group_id] = {'label': label, 'mask': np.zeros((data['imageHeight'], data['imageWidth']), dtype=np.uint8)}

            cv2.fillPoly(masks[group_id]['mask'], [poly_np], 255)
        else:
            # If group_id is not present, treat each polygon separately
            mask = np.zeros((data['imageHeight'], data['imageWidth']), dtype=np.uint8)
            cv2.fillPoly(mask, [poly_np], 255)
            masks[label] = {'label': label, 'mask': mask}

    return list(masks.values())

folder_path = fr'Karachi_4\Labels'
i=0
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.exists(file_path):
        masks = json_to_mask(file_path)
        print(fr"{i} DONE")
        i+=1
        for idx, mask_data in enumerate(masks):
            label = mask_data['label']
            mask = mask_data['mask']
            cv2.imwrite(fr"Karachi_4\Masks\{filename[:-4]}jpg", mask)
    else:
        print(f"{i} The file at {file_path} does not exist.")

# Bismillah