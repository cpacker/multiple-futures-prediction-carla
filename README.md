# Multiple Futures Prediction

This repository accompanies the paper [**Contingencies From Observations (CfD)**]() (ICRA 2021) and implements the Multiple Futures Prediction (MFP) baseline in the paper.
It is based on the public repository [https://github.com/apple/ml-multiple-futures-prediction](https://github.com/apple/ml-multiple-futures-prediction) from the authors of the [MFP paper](https://arxiv.org/abs/1911.00997).

The MFP architecture has been modified from the original repo to not use attention (the scenarios used in CfD only use 2 agents). Additionally, it has been modified to support rotation of model inputs based on the each vehicle's yaw, such that the reference frame (for prediction) always starts at (0,0) in the +X direction. See the [CfD paper appendix]() for further details.

MFP models trained on the CfD dataset can be used for planning, e.g., to control a vehicle in the CARLA simulator. To use an MFP model for control in a CfD scenario, follow the instructions for running MFP models in the [CfD repository]().

### Installation

Apart from CARLA, the install follows the original MFP repo:
```sh
python3.6 -m venv .venv # Create new venv
source ./venv/bin/activate # Activate it
pip install -U pip # Update to latest version of pip
pip install -r requirements.txt # Install everything
```

All experiments were done using CARLA version 0.9.6. Make sure CARLA is installed and can be found in your environment:
```sh
# Downloads hosted binaries.
wget http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz

# CARLA 0.9.6 installation.
tar -xvzf CARLA_0.9.6.tar.gz -C $CARLA_ROOT

# Installs CARLA 0.9.6 Python API.
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg
```

### CARLA Dataset

This repo trains an MFP model on the same dataset used in CfD.

First download the CfD dataset following the instructions in the [CfD repo](https://github.com/JeffTheHacker/ContingenciesFromObservations).

Next, either copy the dataset to the `multiple_futures_prediction` folder, or create a link pointing to the folder:
```sh
cd multiple_futures_prediction
ln -s CFD_DATASET_LOCATION ./carla_dataset_cfd
```

This should create the following directory structure:
```sh
multiple_futures_prediction/carla_data_cfd/Left_Turn_Dataset
multiple_futures_prediction/carla_data_cfd/Right_Turn_Dataset
multiple_futures_prediction/carla_data_cfd/Overtake_Dataset
```

### Usage 

To train a model, run `train_carla_cmd` on a desired config file:
```sh
python -m multiple_futures_prediction.cmd.train_carla_cmd \
--config multiple_futures_prediction/configs/mfp2_carla_rightturn.py
```

To visualize a model, run `demo_carla_cmd` to replay files from training with predictions overlaid with ground truth at each timestep:
```sh
python -m multiple_futures_prediction.cmd.demo_carla_cmd \
--checkpoint-dir CARLA_right_turn_scenario \  # directory with the saved model checkpoint
--outdir mfp2_carla_rightturn_sclean_rotate_nopretrain \  # directory to write the images and video to
--frames 200  # how many frames to include in the video
```

This repo includes saved models for each scenario in the paper, located in `checkpts`:
```sh
python -m multiple_futures_prediction.cmd.demo_carla_cmd --checkpoint-dir CARLA_left_turn_scenario --outdir mfp_carla_leftturn --frames 200
python -m multiple_futures_prediction.cmd.demo_carla_cmd --checkpoint-dir CARLA_right_turn_scenario --outdir mfp_carla_rightturn --frames 200
python -m multiple_futures_prediction.cmd.demo_carla_cmd --checkpoint-dir CARLA_overtake_scenario --outdir mfp_carla_overtake --frames 200
```
