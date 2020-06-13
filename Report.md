# DRLND Project 3 Report (Collaboration and Competition)

## Introduction

FTo solve the environment in this project, I decided to implement the MADDPG algorithm. MADDG is a great choice for this environment due to the existence of multiple agents, and my approach here was inspired by the implementation used in the DRLND Nanodegree Lab for the physical deception environment.

I also added a `DDPG_Tennis.ipynb` solution notebook using the DDPG algorithm I implemented in the previous project for this nanodegree. The reason DDPG works well in this environment, without any changes from what I had implemented for project 2, is that the two agents in the `Tennis` environment are trying to learn the same exact policy and have the same reward structure. In that sense, it is acceptable for the two agents to share a single DDPG instance to learn the best policy possible.

## Learning Algorithm

The [MAPPDG algorithm](https://arxiv.org/pdf/1706.02275.pdf) was introduced as an extension of DDPG for multi-agent environments. One simple way to think of MADDPG is as a wrapper for handling multiple DDPG agents. But the power of this algorithm is that it adopts a framework of centralized training and decentralized execution. This means that there is extra information used during training that is not used during testing. More specifically, the training process makes use of both actors and critics, just like DDPG. The difference is that the input to each agent's critic consists of all the observations and actions for all the agents combined. However, since only the actor is present during testing, that extra information used during training effectively goes away. This framework makes MADDPG flexible enough to handle competitive, collaborative, and mixed environments.

Because of the similarities with DDPG, the hyperparamters used in MADDPG are very similar, and are listed below:

```
BUFFER_SIZE                 # replay buffer size
BATCH_SIZE                  # minibatch size
GAMMA                       # discount factor
TAU                         # for soft update of target parameters
LR_ACTOR                    # learning rate of the actor 
LR_CRITIC                   # learning rate of the critic
WEIGHT_DECAY                # L2 weight decay
UPDATE_EVERY                # weight update frequency
NOISE_AMPLIFICATION         # exploration noise amplification
NOISE_AMPLIFICATION_DECAY   # noise amplification decay
```

The last two hyperparameters were added to amplify the exploration noise used by the agent in an effort to learn the optimal policy faster.

In MADDPG, each agent has its own actor local, actor target, critic local and critic target networks. The model architecture for MADDPG resembles DDPG very closely. The difference is that I found better results when concatenating the states and actions as the input to the critic's first hidden layer. DDPG, on the other hand, concatenates the actions to the input of the second hidden layer, after the states have already gone through in the first hidden layer.

## Training and Results

Training the DDPG solution was very simple. Only a few changes in the hyperparameters from Project 2 solved the environment for me:

![DDPG Plot of Rewards](https://github.com/MarcioPorto/drlnd-collaboration-and-competition/blob/master/ddpg_tennis_training.png)

As you can see above, the training was a little unstable, and the agent never achieved a rolling score above the 0.5 required to solve this environment. I believe I could have stabilized this result, but my main focus in this project was to implement the MADDPG algorithm so decided to move right on.

I found training to be particularly challenging for my final MADDPG implementation. Initially, I had many small issues getting the full implementation to work as expected. After that was worked out, I found the training process to be very sensitive to the hyperparameters. It took a good amount of trial and error to finally get to something that trained well enough:

![MADDPG Plot of Rewards](https://github.com/MarcioPorto/drlnd-collaboration-and-competition/blob/master/maddpg_tennis_training.png)

The results above show that the training process followed a stable pattern, and the 100-episode rolling max score never went significantly down after solving the environment.

One interesting finding from the training step was that my optimal solution did not add any exploratory noise to the agent's chosen action. I found that adding the noise capped the rolling score at 0.05 across multiple runs. I am not quite sure why this is happening, but my best guess is that maybe the noise added was slightly too large and interfered with proper learning. Another possibility is that there is a bug in the code.

Here are the final values for the hyperparameters I ended up using:

```
RANDOM_SEED = 0
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 512
GAMMA = 0.99
TAU = 5e-2
LR_ACTOR = 5e-4
LR_CRITIC = 5e-4
WEIGHT_DECAY = 0.0
UPDATE_EVERY = 2
NOISE_AMPLIFICATION = 1
NOISE_AMPLIFICATION_DECAY = 1
```

## Future Work Ideas

I would like to spend more time playing with the hyperparameters used to see if I can get the agents to achieve the target score for this environment a little faster. I would also like to investigate the exploratory noise situation described above.

Next, I would like to use this implementation to take a crack at the `Soccer` Unity environment, as it presents a more challenging environment where agents can have different reward structures.