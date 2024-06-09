# yolov8-for-satellite-imagery
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohtashimButt/yolov8-for-satellite-imagery/blob/master/main_segmentation.ipynb)
## Overview
By employing remote sensing techniques for data collection followed by the segmentation of green spaces using a fine-tuned deep learning model we performed a temporal and spatial analysis of green spaces in the F7 sector. To evaluate the model’s geographic transferrability, inference was also performed on images of urban centers in Karachi and Swat, two cities in Pakistan with contrasting urban infrastructure and landscapes.

# Methodology
We fine-tuned Yolov8 instance segmentation model and trained it from scratch on satellite imagery of Islamabad's F-7 sector (coordinates: `33.731318, 73.043857` and `33.710294, 73.070434`). 1240 `.png` tiles (each `256x256`) for the entire sector was generated from [GEID](https://www.allmapsoft.com/geid/) at zoom level `20`. 
- **Dataset-1**: Annotation of around 100 tiles to highlight individual greenspace components (e.g. grass, trees, shrubs, etc).
- **Dataset-2**: 200 images were annotated in a way that greenspace components with overlapping boundaries were merged and labeled as a single greenspace. 
![annotations](https://github.com/MohtashimButt/yolov8-for-satellite-imagery/blob/master/Assets/t1.png)
  
We further made augmented versions of each dataset, making 4 version in total as follows:  


Datasets | Training | Validation | Testing | Total |
--- | --- | --- | --- |--- |
 D1_{aug}      | 339 (80%) | 53 (13%)   | 30 (7%) | 422|
 D1_{non-aug}  | 80 (70%)  | 13 (20%)   | 7 (10%) | 100|
 D2_{aug}      | 339 (80%) | 54 (13%)   | 33 (7%) | 426|
 D2_{non-aug}  | 140 (70%) | 40 (20%)   | 20 (10%)| 200|

Some of the visual results for our fine-tuned model as binary masks are given as follows:    
![islo](https://github.com/MohtashimButt/yolov8-for-satellite-imagery/blob/master/Assets/islo.png)


## How to use our work?
- Clone the repository by using
```
git clone https://github.com/MohtashimButt/yolov8-for-satellite-imagery.git
```
- Open the terminal in the `yolov8-for-satellite-imagery` directory and install `requirements.txt` using the following command:
```
pip install -r requirements.txt
```
- Run the following command in the terminal (opened in the same directory):
```
python main_segmentation.py --image_folder [path/to/folder/you/have/your/images/in] --prediction_folder [path/to/folder/you/want/to/save/your/prediction/masks/in] --gt_folder [path/to/folder/you/have/your/ground/truths/in] --IOU [either True or False] --DICE <either True or False>
```
```
- `--image_folder`: it is an `str` value where you'll provide the path to the folder where you've put the satellite images. If you're in the current directory, just give the folder name.
- `--prediction_folder`: it is an `str` value where you'll provide the path to the folder you want your prediction masks to be saved. If you're in the current directory, just give the folder name.
- `--gt_folder`: it is an `str` value where you'll provide the path to the folder you have your ground truths in. If you're in the current directory, just give the folder name.
- `--IOU`: it is a `bool` value. Set is `True` if you want to get the IOU scores for individual prediction and HAVE your ground truths saved in `--gt_folder`
- `--DICE`: it is a `bool` value. Set is `True` if you want to get the DICE Loss for individual prediction and HAVE your ground truths saved in `--gt_folder`
```
> The pre-trained weight can be accessed from [here](https://drive.google.com/file/d/1NLJh5ISEsdUsHF5T9bKKjfpEx6d3ZCcI/view?usp=sharing)
- Download the weight and put it in the same directory.
