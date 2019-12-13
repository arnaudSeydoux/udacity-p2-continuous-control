from unityagents import UnityEnvironment
from ddpg_agent import Agent
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from noises import OUNoise, GaussianNoise
import json

def ddpg(agent, n_episodes=1000, max_t=200, print_every=100):
    # print(f'{num_agents} agents')
    scores_mean_agent = []             # list containing mean scores (over the 20 agents) from each episode
    # scores_mean_last100 = []           # List containing mean value (over the 20 agents) of score_window
    # scores_window = deque(maxlen=10)  # last 100 scores
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        num_agents = len(env_info.agents)
        scores = np.zeros(num_agents) 

        states = env_info.vector_observations
#         print(f'state: {state} {len(state)}')
#         agent.reset()
        for t in range(max_t):
            actions = agent.act(states, False)
            env_info= env.step(actions)[brain_name]
            
            next_states = env_info.vector_observations       # get next state (for each agent)
#             print(f'next_states: {next_states}')
            rewards = env_info.rewards                        # get reward (for each agent)
            dones = env_info.local_done 
            
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            scores += rewards
            if np.any(dones):
                break
        agent.reset()
        # scores_window.append(scores.mean())                # save most recent mean score
        scores_mean_agent.append(scores.mean())            # save most recent mean score
        # scores_mean_last100.append(np.mean(scores_window))
        print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, scores.mean()))
        # print('\rEpisode {}\tScore: {:.2f}'.format(i_episode, scores.mean()), end="")
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            
    return scores_mean_agent

random_seed = mu = theta = sigma = learnings = learning_rates = factor = doClip = network = None


def reset_variables():
    global random_seed , mu , theta , sigma , learnings , learning_rates , factor , doClip , network

    random_seed = 2
    mu = 0.0
    theta= .15
    sigma = .2

    learnings = (5,5)
    # learnings = (10,10)
    learning_rates = (1e-3, 1e-4)
    learning_rates = (1e-3, 1e-4)

    factor = .6
    doClip = True
    network = (256, 128)

def runSimul():
    filename = f'./results/learnings={learnings}, doClip={doClip}, theta={theta}, sigma={sigma}, learning_rates={learning_rates}, network = {network}'
    print(filename)
    noise = OUNoise(brain.vector_action_space_size, random_seed,mu=mu, theta=theta, sigma=sigma)
    # noise = GaussianNoise(brain.vector_action_space_size, factor)
    agent = Agent(state_size=brain.vector_observation_space_size, action_size=brain.vector_action_space_size, random_seed=random_seed, noise=noise, learnings = learnings, learning_rates = learning_rates, network = network, doClip = doClip)
    # scores = ddpg(n_episodes=1000, max_t=300, print_every=100)
    scores= ddpg(agent, n_episodes=100, max_t=10000, print_every=100)

    with open(f'{filename}.json', 'w') as filehandle:
                    json.dump(scores, filehandle)     


env = UnityEnvironment(file_name='reacher20.app')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


reset_variables()
print(f'network: {network}')

runSimul()
# network = (128, 64)
# network = (512, 256)


# decay_rate = .
# for steps in [1,5,10,50]:
#     for times in [1,5,10,50]:
#         filename = f'./results/steps={steps}, times={times}'
#         print(filename)
#         noise = OUNoise(brain.vector_action_space_size, random_seed,mu=mu, theta=theta, sigma=sigma)
#         agent = Agent(state_size=brain.vector_observation_space_size, action_size=brain.vector_action_space_size, random_seed=random_seed, noise=noise, learnings = (steps,times),network = (256, 128), doClip = doClip)
#         # scores = ddpg(n_episodes=1000, max_t=300, print_every=100)
#         scores, _ = ddpg(n_episodes=100, max_t=10000, print_every=100)

#         with open(f'{filename}.json', 'w') as filehandle:
#                         json.dump(scores, filehandle)     





# for learning_rates in [(1e-3, 1e-4), (1e-4, 1e-3),(1e-2, 1e-2),(1e-3, 1e-3),(1e-4, 1e-4),(1e-5, 1e-5)]:
#     runSimul()
# reset_variables()



# for doClip in [True, False]:
#     runSimul()
# reset_variables()

# for learnings in [(1, 1),(5, 5),(10, 10),(20, 20), (30, 30), (10, 5), (20,10), (30,20)]:
#     runSimul()
# reset_variables()


# for network in [(512*2, 256*2),(32, 16),(64, 32),(128, 64),(512, 256)]:
#     runSimul()
# reset_variables()

# for sigma in [1., .8, .5, .2, .1, .01]:
#     runSimul()
# reset_variables()
