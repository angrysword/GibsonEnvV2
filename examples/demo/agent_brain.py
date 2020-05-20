'''
This brain belong to agent/robot. decide what is the best action to taken base on 
1. goal
2. current status
within following context: like environment and current position and so on

'''
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import datasets, layers, models,optimizers
import tensorflow_probability as tfp



BATCH_SIZE=100



class agent_brain(self):
    def __init__(self,is_continue_training):
        #initial or load neural network
        if(is_continue_training):
            pass #load network

        #initiate network
        self.critic=_build_critic_net()
        self.actor=_build_actor_net()

        self.actor.compile(optimizer=optimizers.RMSprop(lr=0.005),loss=self._actor_loss)
        self.critic.compile(optimizer=optimizers.RMSprop(lr=0.005),loss=self._critic_loss)

        
    def suggest_action(self,cur_observation):
        mu,sigma=self.actor.predict(cur_observation)
        sigma=tf.math.exp(sigma)
        action_probs=tfp.distributions.Normal(mu,sigma)
        probs=action_probs.sample([2])
        self.log_probs=action_probs.log_probs(probs)
        action=tf.math.tanh(probs)
        return(action)

        

    def learning_now(self,observe,action,reward,new_observe,done):
        critical_value_=self.critic.predict(new_observe)
        critical_value=self.critic.predict(observe)
        factor=self.gamma*(1-(int)done)
        delta=reward+factor*critical_value_-critical_value

        actor_loss=-self.log_probs*delta
        critic_loss=delta**2

        loss=actor_loss+critic_loss

        #need to know how to backward this
        
        self.actor.fit([observe,delta],action,verbose=0)
        self.critic.fit(observe,reward,verbose=0)

   
    def save_training(self):
        self.training_net.save_weights()
        self.matural_net.set_weights(self.training_net.get_weights())

    def _critic_loss(self):

    def _build_critic_net(self):
        model = models.Sequential()
        #??? one proble for this network, we do not know how to seperate goal
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))

        #no needed activation function. just have two output. one is mean another is standard variation
        model.add(layers.Dense(1)

        

        return(model)

    

    def _build_actor_net(self):
        model = models.Sequential()
        #??? one proble for this network, we do not know how to seperate goal
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))

        #no needed activation function. just have two output. one is mean another is standard variation
        model.add(layers.Dense(2)

        return(model)


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

 

