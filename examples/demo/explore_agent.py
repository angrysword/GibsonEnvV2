import tensorflow as tf

from tensorflow.keras import datasets, layers, models


'''
Use an agent to generate the testing sample to make sure sample is good. 
But do not have a good result so far. because.
1. no loss function
2. do not load learned model

'''
class explore_agent(self):
    def __init__(self):
        self.policy_net=_build_policy_net()
        self.value_net=_build_value_net()

    '''
        explore agent should base on environment return action. but seem it is not enough.
        except action. they also wish we can return value
    
    ---the following is original think
    def next_action(self,obs,goal):
        action=self.explore_net.predict(obs)
        return action




    '''
    def next_action(self,obs,goal):
        action=self.policy_net.predict(obs)
        value=self.value_net.predict(obs)

        return action,value


    #we like to build a neural network to do explore, instead of some 
    
    def _build_policy_net(self):
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

        #??? do not know how to set the loss function.
        model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
        
        return model

     def _build_value_net(self):
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

        #??? do not know how to set the loss function.
        model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
        
        return model

        
    