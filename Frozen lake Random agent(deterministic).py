import gym
import matplotlib.pyplot as plt
import torch
from gym.envs.registration import register   
#Register makes the frozen lake env deterministic
#for example here we will define one of the environment characteristics "is slippery" as false. 

try:
    register(
        id='FrozenLake-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : '4x4', 'is_slippery': False},
        max_episode_steps=100,
        reward_threshold=0.78, # optimum = .8196)
    )
except :
    print("Environment is already registered")
    
def agent():
    
    env=gym.make("FrozenLake-v0")
    
    number_states = env.observation_space.n     #observation space for a 4*4 grids i.e 16states. 
    number_actions= env.action_space.n          #no of actions that can be taken i.e 4 here(North, south, east, west)
    
    print(number_states)
    print(number_actions)
    
    total_steps=[]
    total_rewards=[]
    episodes = 500
    
    # Now we initialise the Q table with zero, where Q is the estimate of the action value. 
    Q=torch.zeros([number_states,number_actions])
    # This basically defines a tensor with zero values.
    # tensor is nothing but an array of number/functions that changes according to certain rule. 
    gamma=0.95
    alpha=0.9
    
    for i in range(episodes):
        state=env.reset()
        
        steps=0
        
        while True:
            
            steps+=1
            
            # Now we choose an action greedily 
            random_values = Q[state] + torch.rand(1,number_actions)/500
            #torch.rand selects a random action out of the total no of actions

            action = torch.max(random_values,1)[1].item()
            # This selects the action with max reward/value 
            
            new_state,reward,done,info=env.step(action)
            
            #Now using the bellman equation we can update the Q table
            
            Q[state,action]=reward+gamma*(torch.max(Q[new_state]))
            
            state=new_state                    #updating the state 
            
            # if the process is done, we will update the total_steps and the total_rewards
            if done:
                total_steps.append(steps)
                total_rewards.append(reward)
                print("The epsisode ended after",steps,"steps")
                break
        env.render()
        env.close()
        env.env.close()
        
    print("Total no of successfull steps is:",sum(total_rewards)/episodes)
    plt.figure(figsize=(12, 5))
    plt.title("Total Rewards")
    plt.bar(torch.arange(len(total_rewards)), total_rewards, alpha=0.6, color='green')
    plt.show()
            
    plt.figure(figsize=(12, 5))
    plt.title("Episod Length")
    plt.bar(torch.arange(len(total_steps)), total_steps, alpha=0.6, color='red')
    plt.show()

agent()
