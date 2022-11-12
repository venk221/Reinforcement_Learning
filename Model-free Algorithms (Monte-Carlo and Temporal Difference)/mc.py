# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.
    You could test the correctness of your code
    by typing 'nosetests -v mc_test.py' in the terminal.

    You don't have to follow the comments to write your code. They are provided
    as hints in case you need.
'''
#-------------------------------------------------------------------------

def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and his otherwise

    Parameters:
    -----------
    observation

    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    # get parameters from observation
    score, dealer_score, usable_ace = observation
    score, dealer_score, usable_ace = observation
    if score>= 20:
        action = 0
    else:
        action = 1
    # action

    ############################
    return action

def mc_prediction(policy, env, n_episodes, gamma = 1.0):
    """Given policy using sampling to calculate the value function
        by using Monte Carlo first visit algorithm.

    Parameters:
    -----------
    policy: function
        A function that maps an obversation to action probabilities
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value

    Note: at the begining of each episode, you need initialize the environment using env.reset()
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> value
    Vdict = defaultdict(float)
    ############################
    # YOUR IMPLEMENTATION HERE #
    for i in range(n_episodes):
    # initialize the episode
        current_state = env.reset()
        # generate empty episode list
        episode = []
        done = False
        # loop until episode generation is done
        while not done:
            # select an action
            action = policy(current_state)
            # return a reward and new state
            observation,reward,done,info,_ = env.step(action)
            # append state, action, reward to episode
            episode.append((current_state,action,reward))
            # update state to new state
            current_state = observation
        G = 0
        states = []
        for index,sample in enumerate(episode):
            # compute G
            state, action, reward = sample   
            if state not in states:
                # print(state)
                G = sum([sample[2]*gamma**i for i,sample in enumerate(episode[index:])])
                states.append(state)
                returns_count[state] += 1
                returns_sum[state] +=G
                Vdict[state] = returns_sum[state]/returns_count[state]      
    ############################

    return Vdict

def epsilon_greedy(Q, state, nA, epsilon = 0.1):
    """Selects epsilon-greedy action for supplied state.

    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1

    Returns:
    --------
    action: int
        action based current state
    Hints:
    ------
    With probability (1 − epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    action = np.ones(nA,dtype=float)*(epsilon/nA)
    BestAction = np.argmax(Q[state])
    action[BestAction] += (1 - epsilon)
    action = np.random.choice(np.arange(nA),p=action)
    ############################
    return action

def mc_control_epsilon_greedy(env, n_episodes, gamma = 1.0, epsilon = 0.1):
    """Monte Carlo control with exploring starts.
        Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-(0.1/n_episodes) during each episode
    and episode must > 0.
    """

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    ############################
    # YOUR IMPLEMENTATION HERE #
    for en in range(n_episodes):
        # define decaying epsilon
        if en!=0:
            epsilon = epsilon - (0.1/n_episodes)  
        # initialize the episode
        ob = env.reset() 
        done = False
        # generate empty episode list
        epi = []
        # loop until one episode generation is done
        while not done:
             # get an action from epsilon greedy policy
            action = epsilon_greedy(Q, ob, env.action_space.n, epsilon)
            # return a reward and new state
            state, reward, done, _ , _  = env.step(action)
            # append state, action, reward to episode
            epi.append((ob,action,reward))
            # update state to new state
            ob = state
        state_action_pair = []
        G = 0
        # loop for each step of episode, t = T-1, T-2, ...,0
        for index,sample in enumerate(epi):
            state,action,reward = sample
            # compute G
            if (state,action) not in state_action_pair:
                G = sum((sample[2]*gamma**i for i,sample in enumerate(epi[index:])))
                state_action_pair.append((state,action))
                returns_count[(state,action)] += 1
                returns_sum[(state,action)] +=G
                Q[state][action] = returns_sum[(state,action)]/returns_count[(state,action)]
    return Q
def main():
    print('Hello World')
    
if __name__ == '__main__':
    main()