[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

# Deep Reinforcement Learning Nanodegree Project 3: Collaboration and Competition

## Introduction

![Trained Agent][image1]

The files in this reporsitory implement a [MADDPG](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf) agent acting in the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment from Unity ML-Agents. In this environment, there are two agents. Each controls a tennis racket and is trying to bounce the ball over a net.

The mechanics of this environment are as follows:

- *Rewards*: An agent receives a reward of +0.1 if it hits the ball over the net. If the agent lets the ball hit the ground or hits the ball out of bounds, it receives a reward of -0.1.
- *State space*: 24 variables describing position and velocity of the ball and racket. Importantly, each agent receives its own local observation. 16 out of the 24 variables correspond to the two previous observations for an agent. That way, you can think of the state space as a stack of 3 observations.
- *Action space*: Vector of 2 numbers corresponding to movement along the x and y axis.

The environment is considered solved, when the average (over 100 episodes) of the agents' scores is at least +0.5. The score for each episode is just the maximum score (without discounting) from either agent.

## Getting Started

1. Clone this repository.

2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

3. Place the downloaded file(s) in the folder you cloned this repo to and unzip (or decompress) the file.

4. Create and activate a Python environment for this project. I recommend using `conda` or `venv`.

5. Activate that environment and install dependencies: 
    ```
    pip install -r requirements.txt
    ```

## Instructions

1. Run `jupyter notebook` and open the `Tennis.ipynb` notebook. 

2. Adjust the path to your desired environment file based on its name and where you placed it.

3. You are ready to start interacting with the environment.
    - Use the cells in sections 1, 2 and 3 to initialize and explore the environment
    - Run the cells in section 4 to train the agent. Feel free to change the hyperparameters in `main.py` to see if you can improve training.
    - Run the cells in section 5 to test the trained agent.