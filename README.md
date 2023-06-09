# RL-on-Elevator-UX
Elevator UX Improvement with RL

## Installation Guide
### Using Python Venv

Clone repo and install [requirements.txt](https://github.com/jentleshin/RL-on-Elevator-UX/blob/main/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

Python 3.10.9 with PyTorch 1.11.0 used for training and testing.

Follow the steps below in order to reproduce our results.

1. Clone repo

```bash
git clone https://github.com/jentleshin/RL-on-Elevator-UX.git  # clone
cd RL-on-Elevator-UX
python -m venv "your_venv_name"
source ./venv/bin/activate
```
2. Install Pytorch

Install <a href="https://pytorch.org/get-started/previous-versions/">Pytorch</a> that matches your CUDA version

Pytorch with CUDA 11.5 used:
```bash
pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 torchaudio==0.11.0+cu115 --extra-index-url https://download.pytorch.org/whl/cu115
```
3. Install Requirements

```bash
pip install -r requirements.txt  # install
```
## Run
### Training 

```bash
python main.py train
#Training with 1000000 timesteps
python main.py train --timesteps 1000000 #default=500000
#Resume training with ./checkpoints/previous_checkpoint.zip
python main.py train --load previous_checkpoint #default=recent

🌟Testing
#Training with new model with discrete buttonState
python main.py train --buttonState passengerNums #default=normal
```

### Testing trained results

```bash
python main.py test 
#Testing with ./checkpoints/previous_checkpoint.zip
python main.py test --checkpoint previous_checkpoint #default=recent
#Testing with 20 episodes
python main.py test --num_episodes 20 #default=10
#Save result video as ./videos/results.mp4
python main.py test --filename results #default=recent

🌟Testing
#Training with new model with discrete buttonState
python main.py test --buttonState passengerNums #default=normal
```

### Testing Baseline Policy

```bash
python main.py baseline
#Testing with 20 episodes
python main.py baseline --num_episodes 20 #default=10
#Save result video as ./videos/baseline_results.mp4
python main.py baseline --filename baseline_results #default=baseline

```

