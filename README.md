# k-armed-Bandit

The multi-armed bandit problem is a classic reinforcement learning problem in which a machine with k number of "arms" compete to maximize overall reward. The problem is analogous to a gambler at a slot machine with k-arms, where each has a hidden true payoff, but when pulled will actually output a reward sampled from a normal distribution centered around this hidden payoff. If each arm rewarded the gambler with its actual payoff, there would be no exploration, and we would always pull this arm. However because this problem is non-stationary (as we are increasing the true value of each arm with a random amount of noise each timestep), the true values continuously fluctuate and this initial arm might not actually be the optimal arm. Therefore, balancing exploration and exploitation is crucial to maximizing the rewards. We have to explore with probability epsilon (in order to occasionally check the payoff of other arms), and the exploit the arm that we believe has the highest value at each timestep (to maximize our projected gains.)


Below is the visualization of the results after running two non-stationary 10-armed testbeds for 10,000 timesteps per episode, and 300 episodes each. The two differ in that one calculates action-value methods using sample averages, the other using a constant step-size parameter (given as alpha = 0.1 in this example.) We can see the constant step-size parameter leading to both higher average rewards as well as higher percentage of optimal action taken in each timestep. 


<img src="./Bandit Results.png" width = "600">
