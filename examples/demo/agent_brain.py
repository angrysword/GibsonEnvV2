'''
This brain belong to agent/robot. decide what is the best action to taken base on 
1. goal
2. current status
within following context: like environment and current position and so on

'''
import numpy as np
import random

BATCH_SIZE=100


class neural_network(self):
    def __init__(self):
        super().__init__()
    pass

class agent_brain(self):
    def __init__(self,is_continue_training):
        #initial or load neural network
        if(is_continue_training):
            pass #load network

        #initiate network
        self.training_net=neural_network()
        self.matural_net=neural_network()

        
    def suggest_action(self,cur_observation):
        multiple_acts_rewards=self.matural_net.predict(cur_observation)
        action=np.argmax(multiple_acts_rewards) #get reward max action

    def learning_now(self,memory):
        batch = np.asarray(random.sample(memory, BATCH_SIZE))

        cur_observes= []
        actions=[]
        max_q_values = []

        #training problem
        # we have training date Xobserve=Yreward with continue action space.
        
        for entry in batch:

            cur_observes.append(entry['cur_observe'])            
            actions.append(entry['action'])
            next_observe=entry['next_observe']

            #multiple actions. think about continue action:https://stackoverflow.com/questions/7098625/how-can-i-apply-reinforcement-learning-to-continuous-action-spaces

            multiple_actions=[]
            multiple_acts_rewards= self.matural_net.predict(next_observe)
            future_reward=np.max(multiple_acts_rewards)


            if entry["terminal"]:
                all_reward= entry["reward"]
            else:
                all_reward= entry["reward"] + GAMMA *future_reward 
            max_q_values.append(all_reward)

            #data have prepared
        
        self.training_net.fit(cur_observes,max_q_values)
    
    def save_training(self):
        self.training_net.save_weights()
        self.matural_net.set_weights(self.training_net.get_weights())



