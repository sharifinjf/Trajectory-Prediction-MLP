<p align="center">
<img src="/Images/video_vehicle_107.png" alt="Alt text" width="800" height="600"/>
</p>
<p align="center">
<img src="/Images/Demo.gif" alt="Alt text" width="800" height="600"/>
</p>



Blue: Represents the past trajectory of the object

Green: Indicates the ground truth path the object followed

Red: Shows the predicted future trajectory based on the model

# Trajectory Prediction Project
## Overview
This project focuses on trajectory prediction using various models. It leverages the KITTI dataset for training and evaluation and provides tools for visualizing the predicted trajectories.
# Setup
## Create a Virtual Environment
To ensure the correct dependencies are installed, it's recommended to create a virtual environment. Run the following commands:


'python -m venv env'

'source env/bin/activate'   # On Windows use `env\Scripts\activate`

## Install Dependencies
Install the necessary Python packages using pip:
pip install -r requirements.txt
## Dataset 
### Download KITTI Dataset
First, download the KITTI Raw dataset and split it into the appropriate format. Place the dataset in the kitti_raw_data folder.
## Training and Evaluation
To train and evaluate your model, run the train.py script. You can choose between different models by setting the appropriate model in the script:
- model_multiLayer
- model_singleLayer
- Or your custom model
Run the script using:
python train.py
## Plotting Trajectories
To visualize the predicted trajectories, run the plot_trajectory.py script:
python plot_trajectory.py
## Pre-trained Models
You can download pre-trained models from this link 
# Repository Structure
train.py: Script to train and evaluate the model.
plot_trajectory.py: Script to plot the predicted trajectories.
kitti_raw_data: Folder where the KITTI dataset should be placed.
# Contributions
Contributions are welcome! Please feel free to submit a Pull Request or open an issue.
