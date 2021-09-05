[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"


# Project: Collaboration and Competition

### Introduction
In this project two agents are trained to play tennis using the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment from the [Unity ml agents toolkit](https://github.com/Unity-Technologies/ml-agents)

![Trained Agent][image1]

The environment comprises of two agents controlling the rackets to hit the ball over the net. An agent is rewarded as follows:
 * Successful hit over the net provides a reward of +0.1
 * Causing the ball to go out of play or hitting ground provides a reward of -0.01
 
Hence the goal of each agent is to play as long as possible without getting out of play (by letting the ball hiiting the ground or hitting the ball hard causing it to go out). It's an episodic task.

##### Observation Sapce
Observation space if a 8 continuous variable vector. Variables stands for the position and velocity of the ball and racket. Both agents have their own local observation vector.
##### Action Sapce
For each agent action is a vector of 2 continuous variables. The variables stands for the movement towards/away from the net and jumping. 
##### Rewards
Reward is a list with 2 entries. Each entry corresponds to the reward received by the two agent.
##### Environment Solution
Solving this environment means the agents to score +0.5 on average over 100 consecutive episodes using the maximum score obtained of the two agents.
- During episode the rewards are added ,undiscounted, to get total reward received by two agents at the end of the episode. Maximum total reward is selected to yield a single **score**.
- This **score** is averaged for last 100 episodes to get **avg_acore**.
- The environment is solved when **avg_score** is at least +0.5.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the same folder as for this Readme.md, and unzip (or decompress) the file. 

### Instructions

The code is divided in the following files
- `buffers.py`: Consists of the code related to the Replay Memory eg. adding and sampling expierences etc.
- `constants.py`: All the constants used are declared in that file eg Replay Memory size etc.
- `utilities.py`: Consists of helper functions eg: converting numpy array to torch tensor etc.
- `networks.py`: Consists of the Neural network classes for actors and critics 
- `agents.py`: Implements the main algorithm for the agent. Multi-Agent DDPG in this case.
- `Tennis.ipynb`: Main file to training the agent. To run the code and train the agent, follow the instructions in this notebook and run the code-blocks

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