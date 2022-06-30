# Reinforcement Learning 

When we think of learning a new task, trial and error is the first thing that strikes our mind. We learn how our sorroundings responds to what we do. For example, while baby is learning to walk , it is aware of how sorroundings (gravity, obstacles ) responds to what steps it take. It seeks to repeat steps that leads it walk longer and avoid steps that makes it fall. Reinforcement Learning is based on this paradigm i.e learning from interaction with sorroudings.

Reinforcement learning( RL ) has enabled machines to learn complicated games like chess. The Elo rating of chess grandmaster Magnus Carlsen, 2864 looks paltry when compared to AlphaZero( machine trained using RL), 4680. That's how powerful reinforcement learning is. 

In this project, I have implemented one of the simplest and widely popular "Q-learning" algorithms to teach a spacecraft to land on a target pad.

# Lunar Landing : 

Problem statement: Space ship starts at the top center with random initial force applied. It needs to be landed between the flags (home) using three engines.

Here the state space ( variables that determine the current position of our spacecraft are : 
- Location of the spacecraft in the horizontal and vertical directions ( x ,y) 
- Linear Velocities in the horizontal(u) and vertical direction (v)
- Angle at which our spacecraft is aligned 
- Angular velocity of our spacecraft 

Possible actions our spacecraft can take are : 
- Fire Left engine 
- Fire right engine 
- Fire Main engine (down engine)
- Do Nothing 

Essentially probelm statement boils down to learning a function, which takes state space as input and outputs which engine to fire so that our spacecraft lands at target pad.

<img src="/images/RL diagram.png" width="400" height="250"/>

# Results : 


## 1.Deep Q learning : 

#### Naive Model ( Before training )
<img src="/images/naive.gif" width="400" height="250"/>

#### Moderately trained Model ( After 400 episodes of training )
<img src="/images/intermediate.gif" width="400" height="250"/>



#### Sufficiently trained model (After 700 episodes of training )
<img src="/images/trained_dqn.gif" width="400" height="250"/>



## Comparision with Other Algorithms 
1. Double Deep Q learning : We see that Double deep Q learning takes almost twice episodes as Deep Q learning to train our model. This is becasue we are training either of the two models in a give episodes, so each model is likely to be half mature as model trained using Deep Q learning.


3. Deouble deep Q learning (updated version) 

### Comparison
<img src="/images/comparison.png" width="400" height="250"/>

### Progression 
<img src="/images/progression.png" width="1000" height="200"/>


# Future Scope

We can explore other algorithms like policy gradient to check if it learns at faster pace than current method. 

We can apply current algrithm to solve more complicated environments. For example amazon organizes AWS DeepRacing League(https://aws.amazon.com/deepracer/) in which we get to train our race car on various tracks and compete with other developers to achieve better lap time


Resources : 
1. https://towardsdatascience.com/double-deep-q-networks-905dd8325412
2. https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/07/DQN-LunarLander.html

