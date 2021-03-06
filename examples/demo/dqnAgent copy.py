'''
I consider this is the agent or robot. the basic function of robot is

Given an environment, goal. robot can eventually reach the goal
To do that, robot first leads to 
1. learning the environment and store what he learned
2. action when the goal is set base on the learning


But let me finished the learning agent first

The current agent can not work on continue function. that is why I work on a new one
'''

import numpy as np
import gym,gym.spaces
from examples.demo.agent_brain import agent_brain

#I am small agent, i can only store so much
MEMORY_LIMIT=5000

BATCH_SIZE = 32

SAMPLES_FOR_TRAINING_THRESHOLD=5000
TRAINING_FREQUENCY = 4
RGET_NETWORK_UPDATE_FREQUENCY = 40000
MODEL_PERSISTENCE_UPDATE_FREQUENCY = 10000
REPLAY_START_SIZE = 50000

#how many times learning to save the model
LEARNING_TO_REMEMBER_THRSHHOLD=10000 
#decide when stop exploration when become older and have too much training already
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_STEPS = 850000
EXPLORATION_DECAY = (EXPLORATION_MAX-EXPLORATION_MIN)/EXPLORATION_STEPS


class dqnAgent(self):

    def __init__(self):

        self.isnewbie=True # a new agent ,never training

        self._set_up_continuous_action_space()
        self.epsilon=EXPLORATION_MAX

        self.brain=agent_brain()


        self.training_memory=[]

        self.learning_experience_count=0

        
        action=self.action_space.sample()
        self.action=np.array([float(action[0]),float(action[1])])
 
        self.reward=0
        self.terminal=False


        
    def set_init_episode_observe(self,observe):
        self.cur_observe=observe

    def next_action(self,current_observe):
        #ramdom action
        if(self.isnewbie or np.random.rand()<self.episolon):
            action=self.action_space.sample()
            action=np.array([float(action[0]),float(action[1])])
            return action 

        #base on learning action
        return brain.suggest_action(current_observe)


    def update(self,observe,reward,done,info):
        new_training_piece=_update_cur_piece(next_observe=observe,reward=reward,terminal=done)
        #store
        self.training_memory.append(new_training_piece)
        if len(self.training_memory) > MEMORY_LIMIT:
           self.training_memory.pop(0)

        #status reset
        self.cur_observe=observe

        #if agent have enough memory, put out one batch training
        if len(self.training_memory)>SAMPLES_FOR_TRAINING_THRESHOLD:
            _reflect_learning() #base on memory learning something



    def save_training(self):
        #wish save summary
        self.brain.save_training()
        
    def _update_cur_piece(self,cur_observe,next_observe,reward,terminal):
        training_piece={"current_state": self.cur_observe,
                            "action": self.action,
                            "reward": self.reward,
                            "next_observe": self.next_observe,
                            "terminal": self.terminal}
        return training_piece
    
    def _reflect_learning(self):
        #let braining training
        self.brain.learning_now()

        self._update_epsilon()
        self.learning_experience_count=+1

        if self.learning_experience_count% LEARNING_TO_REMEMBER_THRSHHOLD== 0:
            self._save_training() #save training and also update brain

        #after training, agent have more experience, no need to explore too much
        self._update_epsilon()
        self.isnewbie=False

 
    def _set_up_continuous_action_space(self):
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)

        if self.action_high is not None and self.action_low is not None:
            self.action_high = np.full(shape=self.action_dim, fill_value=self.action_high)
            self.action_low = np.full(shape=self.action_dim, fill_value=self.action_low)
        else:
            self.action_high = np.full(shape=self.action_dim, fill_value=self.velocity)
            self.action_low = -self.action_high
    def _update_epsilon(self):
        self.epsilon -= EXPLORATION_DECAY
        self.epsilon = max(EXPLORATION_MIN, self.epsilon)
