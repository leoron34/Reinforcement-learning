import gym
def main():
    env=gym.make("CartPole-v1")                      #Starts the environment 
    
    total_steps=[]                                   #variable to count the no of steps in each episodes
    episodes=100                                     #defining the no of episodes
    
    for i in range(episodes):
        env.reset()                                  #Starts a new episode
        
        steps = 0
        # now the loop to proceed through the steps taken in a single episode
        while True:
         steps += 1
         action = env.action_space.sample()           #takes a random action
        
         new_state,reward,done,info = env.step(action)  #according to that action, a new state, reward and environment is given. 
         print(new_state)
         print(info)
        
        if done:
            total_steps.append(steps)
            print("Episode finished after {} steps".format(steps))
            break
            
    env.close()

main()
