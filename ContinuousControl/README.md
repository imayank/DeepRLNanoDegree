[//]: # (Image References)

[image1]: ./sample_play.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"



# Project: Continuous Control

### Introduction
In this project an agent is trained to control a double jointed arm.  [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment from the [Unity ml agents toolkit](https://github.com/Unity-Technologies/ml-agents) is used for training.

![Trained Agent][image1]

The environment comprises of a single agent controlling a double jointed arm to reach a target location. Agent is rewarded as follows:
* If the arm reaches the target location an agent receives +0.1 as reward. At each step when the arm is in target location +0.1 is provided to the agent.
* If the arm doesn't reach the target location a reward of 0 is received.

Hence the goal of an agent is to control the arm in such a way that it maintains its position at target location as long as possible. Maximum time-steps in an episode is 1000. 

##### Observation Sapce
Observation space is a vector of 33 continuous variables. Variables stands for the position, rotation, velocity, and angular velocities of the arm.
##### Action Sapce
Action is a vector of 4 numbers. The variables stands for the torque applied to two joints. Every entry in the action vector is a number between -1 and 1.

##### Environment Solution
Solving this environment means that the agent needs to obtain an average score of +30 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the same folder as for this Readme.md, and unzip (or decompress) the file. 

### Instructions

The code is divided in the following files
- `buffers.py`: Consists of the code related to the Replay Memory eg. adding and sampling expierences etc.
- `segment_tree.py`: It implements the segment tree data structure.
- `constants.py`: All the constants used are declared in that file eg Replay Memory size etc.
- `utilities.py`: Consists of helper functions eg: converting numpy array to torch tensor etc.
- `networks.py`: Consists of the Neural network classes for actors and critics 
- `agents.py`: Implements the main algorithm for the agent. DDPG in this case.
- `ReacherSingleAgent.ipynb`: Main file to training the agent. To run the code and train the agent, follow the instructions in this notebook and run the code-blocks

### Dependencies

Before running the code for training and/or watching the trained agent play you need to setup a python environment. Assuming Anaconda/Miniconda is installed on the system follow the instrauctions to setup the environment:

1. Create (and activate) a new environment with Python 3.6.

- __Linux__ or __Mac__: 
   ```bash
   conda create --name drlnd python=3.6
   source activate drlnd
   ```
- __Windows__: 
  ```bash
  conda create --name drlnd python=3.6 
  activate drlnd
  ```
2. Clone the repository (if you haven't), and navigate to the `python/` folder. Once inside the pyton folder run the following command to install the dependecies
   ```bash
   pip install .
   ```

3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
   ```bash
   python -m ipykernel install --user --name drlnd --display-name "drlnd"
   ```

4. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

   ![Kernel][image2]