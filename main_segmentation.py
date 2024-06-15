# %% [markdown]
# ## Before you start
# 
# Let's make sure that we have access to GPU. We can use `nvidia-smi` command to do that. In case of any problems navigate to `Edit` -> `Notebook settings` -> `Hardware accelerator`, set it to `GPU`, and then click `Save`.

# %%
import cv2
import numpy as np
import pandas as pd
# from google.colab.patches import cv2_imshow
# from google.colab import drive
import argparse
# drive.mount('/content/drive')
parser = argparse.ArgumentParser(description="Script to generate masks from satellite imagery")
parser.add_argument('--image_folder', type=str, default="Images", help='The folder where you have put the satellite images')
parser.add_argument('--prediction_folder', type=str, default="Prediction", help='The folder where you want your masks to be saved')
parser.add_argument('--gt_folder', type=str, default="Ground_Truth", help='Add here the name of the folder where you have your ground truths saved')
parser.add_argument('--IOU', type=bool, default=False, help='set it True if you want to generate the IOU scores and have ground truths.')
parser.add_argument('--DICE', type=bool, default=False, help='set it True if you want to generate the DICE loss and have ground truths.')

args = parser.parse_args()

folder_path = args.image_folder
prediction_path = args.prediction_folder + "/"
ground_truth_folder_path = args.gt_folder + "/"
IOU_decision = args.IOU
DICE_decision = args.DICE


# %%
# !nvidia-smi

# %%
import os
HOME = os.getcwd()
print(HOME)

# %% [markdown]
# ## Install YOLOv8
# 
# ⚠️ YOLOv8 is still under heavy development. Breaking changes are being introduced almost weekly. We strive to make our YOLOv8 notebooks work with the latest version of the library. Last tests took place on **12.02.2012** with version **YOLOv8.0.28**.
# 
# If you notice that our notebook behaves incorrectly - especially if you experience errors that prevent you from going through the tutorial - don't hesitate! Let us know and open an [issue](https://github.com/roboflow/notebooks/issues) on the Roboflow Notebooks repository.
# 
# YOLOv8 can be installed in two ways - from the source and via pip. This is because it is the first iteration of YOLO to have an official package.

# %%
# Pip install method (recommended)

# !pip install ultralytics

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

# %%
# Git clone method (for development)

# %cd {HOME}
# !git clone github.com/ultralytics/ultralytics
# %cd {HOME}/ultralytics
# !pip install -e .

# from IPython import display
# display.clear_output()

# import ultralytics
# ultralytics.checks()

# %%
from ultralytics import YOLO

from IPython.display import display, Image

# %% [markdown]
# ## CLI Basics

# %% [markdown]
# If you want to train, validate or run inference on models and don't need to make any modifications to the code, using YOLO command line interface is the easiest way to get started. Read more about CLI in [Ultralytics YOLO Docs](https://docs.ultralytics.com/usage/cli/).
# 
# ```
# yolo task=detect    mode=train    model=yolov8n.yaml      args...
#           classify       predict        yolov8n-cls.yaml  args...
#           segment        val            yolov8n-seg.yaml  args...
#                          export         yolov8n.pt        format=onnx  args...
# ```

# %% [markdown]
# ## Inference using a pre-trained model
# Put your images in the `Images` (in `.jpg` format) folder and run the cell below

# %%
def get_mask(predict, filename, tag):
    for result in predict:
      if result.masks == None:
        print(filename,"in",tag, "is MASLA")
        return np.zeros((256, 256), dtype=np.uint8)
        break
      masks = result.masks.cpu().numpy()
    index = (len(masks))
    hey = 0
    BIG_LIST = list()

    for i in range(index):
      if predict[0].masks == None:
        break
      a = (predict[0].masks.data[i].cpu().numpy() * 255).astype("uint8")
      hey+=a
    return hey
def calculate_white_percentage(mask):
    white_pixels = np.sum(mask != 0)
    total_pixels = mask.size
    percentage = (white_pixels / total_pixels) * 100
    return percentage

model = YOLO("Dataset1_NON_AUG.pt")
# folder_path = "Images"
with open(os.path.join(prediction_path[:-1], "percentage_greenspace.txt"), "w") as file:
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.exists(file_path):
            predict_1_NON_AUG = model.predict(source=file_path, save=False, save_txt=False, boxes=False)
            masks = get_mask(predict_1_NON_AUG, filename, "DATA_1_NON_AUG")
            cv2.imwrite(fr"{prediction_path}{filename[:-4]}.jpg",masks)
            print(filename,f" has been saved to {prediction_path} folder")
            
            white_percentage = calculate_white_percentage(masks)
            file.write(f"{white_percentage:.2f}% of {filename} is greenspace\n")
            print(f"{white_percentage:.2f}% of {filename} is greenspace")

# %% [markdown]
# ## IOU Score
# If you have ground truth masks for the images you want inferences of, you can put them in `Ground_Truth` folder and run the following cell.
# 
# Note: Don't forget to create a `IOU.txt` in the `Eval_Metrices` folder

# %%
def calculate_iou(ground_truth, prediction):

    if not (os.path.exists(prediction)):
      print("inference mask doesn't exist")
    # Read binary masks
    mask_gt = cv2.imread(ground_truth, cv2.IMREAD_GRAYSCALE)
    mask_pred = cv2.imread(prediction, cv2.IMREAD_GRAYSCALE)

    # Threshold the masks to ensure binary values
    _, mask_gt = cv2.threshold(mask_gt, 128, 255, cv2.THRESH_BINARY)
    _, mask_pred = cv2.threshold(mask_pred, 128, 255, cv2.THRESH_BINARY)

    # Compute intersection and union
    intersection = np.logical_and(mask_gt, mask_pred)
    union = np.logical_or(mask_gt, mask_pred)

    # Calculate IoU
    if np.sum(union) == 0:
      return np.sum(intersection) / (np.sum(mask_gt)+0.0000007)
    iou = np.sum(intersection) / (np.sum(union)+0.0000007)
    return iou


ground_truth_folder_path = fr'Ground_Truth/'
# prediction_path = fr"Masks/"

if IOU_decision:
    iou_score_list = list()
    for i in os.listdir(ground_truth_folder_path):
        iou_score = calculate_iou(ground_truth_folder_path+i, prediction_path+i)
        with open('Eval_Metrices/IOU.txt', 'a') as file:
            file.write(f"{i}--> IOU SCORE: {iou_score}\n")
        print(i)
        if iou_score!=0:
            iou_score_list.append(iou_score)

    with open('Eval_Metrices/IOU.txt', 'a') as file:
        file.write(f"Arithmatic mean IoU score: {sum(iou_score_list)/len(iou_score_list)}")
    print(f"Your IOUs have successfully been saved to IOU.txt file in Eval_Metrices folder\nArithmatic Mean IoU score for all images is: {sum(iou_score_list)/len(iou_score_list)}")


# %% [markdown]
# ## DICE Loss
# If you have ground truth masks for the images you want inferences of, you can put them in `Ground_Truth` folder and run the following cell.
# 
# Note: Don't forget to create a `DICE.txt` in the `Eval_Metrices` folder

# %%
def calculate_dice_loss(ground_truth, prediction):
    # Read binary masks
    mask_gt = cv2.imread(ground_truth, cv2.IMREAD_GRAYSCALE)
    mask_pred = cv2.imread(prediction, cv2.IMREAD_GRAYSCALE)

    # Threshold the masks to ensure binary values
    _, mask_gt = cv2.threshold(mask_gt, 128, 255, cv2.THRESH_BINARY)
    _, mask_pred = cv2.threshold(mask_pred, 128, 255, cv2.THRESH_BINARY)

    # Normalizing
    mask_gt = 1/255*mask_gt
    mask_pred = 1/255*mask_pred

    # Compute intersection and union
    intersection = np.logical_and(mask_gt, mask_pred)
    sum_add = np.sum(mask_gt) + np.sum(mask_pred)

    # Calculate IoU
    dice = 2 * np.sum(intersection) / (sum_add+0.0000007)
    return 1-dice

ground_truth_folder_path = fr'Ground_Truth/'
# prediction_path = fr"Masks/"

if DICE_decision:
    dice_loss_list = list()
    for i in os.listdir(ground_truth_folder_path):
        dice_loss = calculate_dice_loss(ground_truth_folder_path+i, prediction_path+i)
        with open('Eval_Metrices/DICE.txt', 'a') as file:
            file.write(f"{i}--> DICE LOSS: {dice_loss}\n")
        print(i)
        if dice_loss!=1.0:
            dice_loss_list.append(dice_loss)

    with open('Eval_Metrices/DICE.txt', 'a') as file:
        file.write(f"Arithmatic mean DICE Loss: {sum(dice_loss_list)/len(dice_loss_list)}")
    print(f"Your DICE loses have successfully been saved to DICE.txt file in Eval_Metrices folder\nArithmatic Mean DICE LOSS for all images is: {sum(dice_loss_list)/len(dice_loss_list)}")