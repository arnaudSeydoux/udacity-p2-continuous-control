[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"




## Introduction

This project trains a reinforcement learning agent to play a game.

We use for the project the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.


## Getting Started

1. You should have python 3.7 installed on your machine.

2. Clone the repo into a local directory

    Then cd in it:
    `cd p2-continuous-control`


3. To install the dependencies, we advise you to create an environment.
    If you use conda, juste run:
    `conda create --name p2cc python=3.7`
    to create `dqn` environment

    Then install requirements files:
    `pip install -r requirements.txt`

4. Run the training file to train the agent, and see progress in training along time
`python train.py`

5. You'll find more infos on the project in `Report.md` file.




