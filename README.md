[//]: # (Image References)

# Project 1: Navigation

### Introduction

This project is the first project in the [Udacity Deep Reinforcement Learning Nano Degree](https://www.udacity.com/courses/deep-reinforcement-learning-nanodegree--nd893)

For this project, we will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent](images/navigation.gif)

A reward of +1 is provided for collecting a yellow banana, a reward of -1 is provided for collecting a blue banana.
And 0 otherwise! 
Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.


### Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

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
2. clone this repository and install the requirements in the python folder with `pip install ./python`
3. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

4. unzip the folder and make sure to point to it in the Navigation.ipynb in the Cell under Step 2. Setting up the environment
By default I assume it is under `/data/Banana_Linux_NoVis/Banana.x86_64`, but you can set anywhere as long as you point to it like in the line `env = UnityEnvironment(file_name="/data/Banana_Linux_NoVis/Banana.x86_64")`

5. Run the cells of the notebook in order 

### Instructions

Follow the instructions in `Navigation.ipynb` to get started with training an agent!  

Feel free to explore the trained Agent by running the notebook `play_trained_agent.ipynb`
