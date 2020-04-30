'''
This brain belong to agent/robot. decide what is the best action to taken base on 
1. goal
2. current status
within following context: like environment and current position and so on

'''
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import datasets, layers, models



BATCH_SIZE=100



class agent_brain(self):
    def __init__(self,is_continue_training):
        #initial or load neural network
        if(is_continue_training):
            pass #load network

        #initiate network
        self.training_net=_build_net()
        self.matural_net=_build_net()

        
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

    def _build_net(self):
        model = models.Sequential()
        #??? one proble for this network, we do not know how to seperate goal
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))


        #??? tanh may not be a good activation function
        model.add(layers.Dense(2,activation='tanh')


        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Policy loss
        neglogpac = train_model.pd.neglogp(A)
        # L = A(s,a) * -logpi(a|s)
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # Value loss
        vf_loss = losses.mean_squared_error(tf.squeeze(train_model.vf), R)

        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef



        #??? do not know how to set the loss function.
        model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
        
        return model

 

