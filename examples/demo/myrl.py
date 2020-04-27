
'''
Goal: work out an algorithm which can navigate from start point to goal. this is a training program
Solution: Use Atari Method first.
Step 1: initialize the basic parameter
        a1. reward function. how to design a reward function.
        a2. Memory size and training batch threshhold, 
        env, state, action space have already
Step 2: experient and training
        b1. run a serial random steps, no mather reach Terminal state. when reach terminal state, restart over.
        b2. design a policy network, when memory size reached and training it and update network size. and base on espisolon to decide use 

Code Architecture
1. Main program: loop until it reach the final steps. terminated.
2. Nerual Network for policy value function



'''
#No matter which one reach the largest and stop training
TOTAL_EPISODE=100
TOTAL_STEPS=50000

LARGEST_STEPS_PER_EPISODE=600

from gibson2.envs.locomotor_env import NavigateEnv, NavigateRandomEnv
from time import time
import numpy as np
from time import time
import gibson2
import os
from examples.demo.dqnAgent import dqnAgent

#the following is actually a training
if __name__ == "__main__":
    config_filename = os.path.join(os.path.dirname(gibson2.__file__),
                                   '../examples/configs/turtlebot_p2p_nav_house.yaml')
    nav_env = NavigateRandomEnv(config_file=config_filename, mode='gui')

    agent=dqnAgent()

    total_steps=0
    for j in range(TOTAL_EPISODE):
        

        if total_steps>TOTAL_STEPS:
            print("Finished largest steps: %d, but Only Finished episods/Total: %d/%d "%(TOTAL_STEPS,j,TOTAL_EPISODE))
            #wish save the training result
            print("training finished")
            #wish have some training results save.
            agent.save_training()
            break

        observe=nav_env.reset()
        agent.set_init_episode_observe(observe)

        nav_env.current_step()

        
        for i in range(LARGEST_STEPS_PER_EPISODE):    # 300 steps, 30s world time

            
            action = agent.next_action(observe)

            observe,reward,done,info = nav_env.step(action)

            agent.update(observe,reward,done,info)

            if done:
                break

    nav_env.clean()
