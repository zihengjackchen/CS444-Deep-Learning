## Deep Reinforcement Learning
### Spring 2023 CS444 Assignment 5

[Assignment Page](https://slazebni.cs.illinois.edu/spring23/assignment5.html)

#### Description
In this assignment, I've been tasked with implementing the famous Deep Q-Network (DQN) and its successor Double DQN to play the game of Atari Breakout using the OpenAI Gym. The assignment aims to help me (1) understand how deep reinforcement learning operates when dealing with pixel-level data from an environment and (2) create a recurrent state to encode and maintain a history of actions and observations.

I've downloaded the initial code provided for this assignment to get started.

To guide me through the implementation, there's a top-level notebook named "MP5.ipynb." It will walk me through all the steps of implementing DQN and Double DQN. My primary focus will be on training the agent in the "agent.py" file for DQN and "agent_double.py" for Double DQN. It's important not to modify the neural network architecture since it's crucial for grading consistency.

Due to computational limitations, the expected performance is to achieve a mean score of 10, which should take around 2000 episodes. The hyperparameters are defined in "config.py," and I have the option to experiment with them, but the provided parameters should be sufficient to reach the target score.

In the context of this assignment, one episode corresponds to a game played by the agent until it exhausts all its lives (in this case, the agent has 5 lives). It's worth noting that the paper's definition of an episode involves almost 30 minutes of training on a GPU, which is not feasible for this assignment.

To assist in debugging and monitoring progress, here's an example of expected rewards versus the number of episodes:

200 episodes: 1.5
400 episodes: 1.5
600 episodes: 1.5
1000 episodes: 1.75
1200 episodes: 2.5
1400 episodes: 3.5
1600 episodes: 5.0
My goal is to ensure that either "agent.py" or "agent_double.py" reaches an evaluation score of 10. To ensure a fair comparison in the report, I'm expected to run both files for the same number of episodes, but only one model needs to achieve the target evaluation score of 10.

#### Top level file:
`./MP5.ipynb`
