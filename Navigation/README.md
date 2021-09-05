[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

# Project: Navigation

### Introduction
In this project an agent is trained to navigate in a large square environment and collect bananas. For this purpose the Banana collector environment of [Unity ML agents](https://github.com/Unity-Technologies/ml-agents) is used.

![Trained Agent][image1]

The environment has banana's of two colors: blue and yellow. An agent is rewarded as follows for collecting a banana:
- If an agent collects a yellow banana a reward of +1 is received
- If an agent collects a blue banana a reward of -1 is received
- otherwise no reward or reward of zero is received

Hence, an agent's goal is to collect as many yellow bananas as possible while avoiding blue bananas. It's an episodic task.

##### Observation Space
State of an agent is 37 dimensional vector and it comprises of velocity of an agent and ray-based perception of objects around agent's forward direction.
##### Action Space
Action space for the environment is discrete. Four discrete actions are available:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

##### Environment Solution 
Solving this environment means that the agent needs to obtain an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the same folder as for this Readme.md, and unzip (or decompress) the file.

### Instructions
The code is divided in the following files
- `model.py`: Consists of the Neural network class for deep Q network 
- `ddqn_agent.py`: Implements the main algorithm for the agent. Deep reinforcement learning with double Q learning.
- `banana_dqn.ipynb`: Main file for training the agent. To run the code and train the agent, follow the instructions in this notebook and run the code-blocks

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
