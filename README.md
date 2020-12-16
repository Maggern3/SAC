## Soft Actor Critic on UnityML's Obstacle Tower Environment
### Project environment
A procedurally generated environment to challenge the state of the art algorithms in planning and generalization. In other words, the environment changes every time the agent sees it. Requiring an agent to truly generalize it's policies in order to succeed. Soft Actor critic has been shown to generalize well in highly complex tasks.
The solution uses computer vision to go from pixels as inputs to actions as outputs.

The possible actions has 4 dimensions;

0. Movement (No-Op/Forward/Back)
1. Camera Rotation (No-Op/Counter-Clockwise/Clockwise)
2. Jump (No-Op/Jump)
3. Movement (No-Op/Right/Left) 

The environment takes these in as a Numpy array with 4 elements where a number specifies the index of the action. The agent can perform one action in each of the 4 action dimensions at the same time.

### Installation 
First clone this repo.
Then run
```
cd obstacle_tower_env
pip install -e .
```
this installs the required dependencies(tested on python 3.6.8 64-bit).

You should now be able to build and run the project.

### Training the agent
Run the following command to train the agent
```
python runner.py
```