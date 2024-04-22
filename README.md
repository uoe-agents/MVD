# Multi-view Disentanglement

This is the implementation of Multi-view Disentanglement (MVD) from the paper Multi-view Disentanglement for Reinforcement Learning with Multiple Cameras.

This code is based on the DrQ PyTorch implementation by [Yarats et al.](https://github.com/denisyarats/drq) 
and the DMControl Generalisation Benchmark by [Hansen et al.](https://github.com/nicklashansen/dmcontrol-generalization-benchmark). As per the original code bases, 
we use [kornia](https://github.com/kornia/kornia) for data augmentation.

The MVD auxiliary task applied to the base RL algorithm is largely contained in the `algorithms/multi_view_disentanglement.py` 
file. The `metaworld` folder contains the [MetaWorld](https://github.com/Farama-Foundation/Metaworld) code with a small change to create the camera angles. Our InfoNCE loss implementation is adapted from [info-nce-pytorch](https://github.com/RElbers/info-nce-pytorch).

## Requirements
We assume you have access to [MuJoCo](https://github.com/openai/mujoco-py) and a GPU that can run CUDA 11.8. 
Then, the simplest way to install all required dependencies is to create a conda environment by running:
```(python)
conda env create -f conda_env.yml
```
You can activate your environment with:
```(python)
conda activate multi_view_disentanglement
```

## Instructions
You can run the code using the configuration specified in `arguments.py` with:
```(python)
python train.py
```

The `configs` folder contains bash scripts for all the algorithms used in the paper 
on the Panda Reach and MetaWorld Soccer tasks as examples. You can run a specific configuration using the 
bash script, for example:
```(python)
sh configs/panda_reach_sac_mvd.sh
```

This will produce the `runs` folder, where all the outputs are going to be stored including train/eval logs.


The console output is also available in the form:
```
| train | E: 5 | S: 5000 | R: 11.4359 | SR: 0.0 | D: 66.8 s | BR: 0.0581 | ALOSS: -1.0640 | CLOSS: 0.0996 | TLOSS: -23.1683 | TVAL: 0.0945 | AENT: 3.8132 | RECONLOSS: 0.7837 | MVDLOSS: 0.6953
```
a training entry decodes as
```
train - training episode
E - total number of episodes
S - total number of environment steps
R - episode return
SR - success rate
D - duration in seconds
BR - average reward of a sampled batch
ALOSS - average loss of the actor
CLOSS - average loss of the critic
TLOSS - average loss of the temperature parameter
TVAL - the value of temperature
AENT - the actor's entropy
RECONLOSS - average image reconstruction loss (for SAC with image reconstruction)
MVDLOSS - average multi-view disentanglement loss
```
An evaluation entry on all training cameras
```
| eval  | E: 20 | S: 20000 | R: 10.9356 | SR: 0.0
```
contains
```
E - evaluation was performed after E episodes
S - evaluation was performed after S environment steps
R - average episode return computed over `num_eval_episodes` (usually 10)
SR - average episode success rate computed over `num_eval_episodes` (usually 10)
```
An evaluation on each individual camera (when `eval_on_each_camera=True`) contains
```
| eval_scenarios  | E: 20 | S: 20000 | CAM1R: 13.2467 | CAM3R: 3.8526 | CAM1SR: 0.0 | CAM3SR: 0.0
```
contains
```
E - evaluation was performed after E episodes
S - evaluation was performed after S environment steps
CAM1R - average episode return computed over `num_eval_episodes` (usually 10) using only the first-person camera
CAM3R - average episode return computed over `num_eval_episodes` (usually 10) using only the third-person camera
CAM1SR - average episode success rate computed over `num_eval_episodes` (usually 10) using only the first-person camera
CAM3SR - average episode success rate computed over `num_eval_episodes` (usually 10) using only the third-person camera
```