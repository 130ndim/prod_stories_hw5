### Run example

1. Install requirements:
`pip install -r requirements.txt`

2. Clone submodules
`git submodule update --remote`

3. Run training
`python ppo_example.py`
   
### Model
A simple CNN

3 conv layers (16, 32, 32)

2 layer output FFN (32, 32)
   
### Reward
The reward is a normalized sum of:
1. 10 * new_explored
2. +1 if moved and the cell is new else -1
3. +1 if the cell is new else -1
4. -1 step penalty

### Dashboard

[WandB log](https://wandb.ai/leonov/vac/runs/765rxorj/overview?workspace=user-leonov) that contains hyperparameters, plots and .gif files with running agent.

### Future steps
1. To make the model see previous frames. (E.g. 3D-CNN, CNN-LSTM)
2. To include custom metrics. (E.g. repetition rate)
3. To support more RL algorithms.