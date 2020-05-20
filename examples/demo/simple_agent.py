
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras import Model,optimizers
from tensorflow.compat.v1 import distributions
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import numpy as np
import gym
import  logging

#physical_devices = tf.config.list_physical_devices('GPU') 
#tf.config.experimental.set_memory_growth(physical_devices[0], True)



ENTROPY_BETA=1e-4

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__('mlp_policy')
        # no tf.get_variable(), just simple Keras API
        self.hidden1 = kl.Conv2D(64,(3,3),1,activation='relu')
        self.hidden2 = kl.Dense(32, activation='relu')
        self.flatten=kl.Flatten()
        
        self.value = kl.Dense(1, name='value')
        
        
        # logits are unnormalized log probabilities
        self.action= kl.Dense(2,activation='relu', name='policy_logits')
        self.sigma= kl.Dense(2, activation='softplus',name='policy_sigma')

    def call(self, inputs):
        # inputs is a numpy array, convert to Tensor
        #x = tf.convert_to_tensor(inputs)
        # separate hidden layers from the same input tensor
        x=inputs
        xa=self.hidden1(x)
        xa=self.hidden2(xa)
        xa=self.flatten(xa)
        action=self.action(xa)


        xv= self.hidden1(x)
        xv=self.hidden2(xv)
        xv=self.flatten(xv)
        value=self.value(xv)
        
        xs=self.hidden1(x)
        xs=self.hidden2(xs)
        xs=self.flatten(xs)
        sigma=self.sigma(xs)
        
        return action,sigma,value 

    def action_value(self, obs):
        # executes call() under the hood
        action,sigma,value= self.call(obs)
        #sigma have problem too        
        sigma=tf.nn.softplus(sigma)+1e-5
        
        norm_dist=distributions.Normal(action,sigma)

        action_va=norm_dist.sample(1)

        neglogav=-norm_dist.log_prob(action_va)

        #neglogav=-tf.math.log(norm_dist.prob(action_va))

        return action_va,neglogav,sigma,value

class A2CAgent:
    def __init__(self, model):
        # hyperparameters for loss terms, gamma is the discount coefficient
        self.params = {
            'gamma': 0.99,
            'value': 0.5
        }
        self.model = model
        self.model.compile(
            optimizer=ko.RMSprop(lr=0.0007),
            # define separate losses for policy logits and value estimate
            loss=[self._all_losses,None,self._value_loss]
        )
    
    def train(self, env, batch_sz=32, updates=1000):
        # storage helpers for a single batch of data
        actions = np.empty((batch_sz,2) )
        neglogavs= np.empty((batch_sz,2) )
        sigmas= np.empty((batch_sz,2) )
        
        rewards, dones, values,refer_next_state_values = np.empty((4, batch_sz))
        
        observations = np.empty((batch_sz,256,256,3))
        # training loop: collect samples, send to optimizer, repeat updates times
        ep_rews = [0.0]
        next_obs = env.reset()
        
        for update in range(updates):

            #run a batch of sample    
            for step in range(batch_sz):
                next_obs=next_obs['rgb']/255.0
                observations[step] = next_obs.copy()
                
                actions_,neglogav_,sigma_, values_ = self.model.action_value(next_obs[None, :])
                _,_,refer_v=self.model.predict(next_obs[None,:])

                #actions[step],neglogavs[step],sigmas[step], values[step] = actions_[0,0,:],neglogav_[0,0,:],values_
                actions[step],neglogavs[step],sigmas[step], values[step] = np.squeeze(actions_),np.squeeze(neglogav_),np.squeeze(sigma_),values_
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])


                ep_rews[-1] += rewards[step]

                refer_next_state_values[step]=tf.stop_gradient(refer_v)

                if dones[step]:
                    ep_rews.append(0.0)
                    next_obs = env.reset()
                    logging.info("Episode: %03d, Reward: %03d" % (len(ep_rews)-1, ep_rews[-2]))


            temp_obs=next_obs['rgb']/255.0
            _,_, next_value = self.model.predict(temp_obs[None,:])
            refer_next_state_values=np.append(refer_next_state_values,tf.stop_gradient(next_value))
            #refer_next_state_values=tf.stop_gradient(refer_next_state_values)
            returns=np.empty(batch_sz)
            for t in reversed(range(rewards.shape[0])):
                returns[t] = rewards[t] + self.params['gamma'] * refer_next_state_values[t+1] * (1-dones[t])
            #returns = returns[:-1]
            # advantages are returns - baseline, value estimates in our case
            advantages = returns - values

            policy_loss=tf.math.reduce_mean(neglogavs)*advantages  #i dont use mean. I suppose use mean base on baseline

            entropy_loss=ENTROPY_BETA*(-tf.reduce_mean((tf.math.log(2*np.pi*sigmas)+1)/2)) # it is different with baseline
            value_loss=tf.keras.losses.mean_squared_error(returns,values)

            #policy loss  base on delta theta log(pi(a|@s)): pi(a|@s)=N(u(@s),sigma**2), s was represented with neural network
            #log(pi(a|s))= (a-u(s))/sigma**2 * @s
            all_losses=self._compute_all_loss(policy_loss,entropy_loss,value_loss)

            #print(self._value_loss(returns,values))
            #losses = self.model.train_on_batch(observations, [all_losses,sigmas,returns])
            losses = self.model.train_on_batch(observations, [all_losses,returns])
            print(losses)
            logging.debug("[%d/%d] Losses: %s" % (update+1, updates, losses))
        return ep_rews

    def test(self, env, render=False):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action, _ = self.model.action_value(obs[None, :])
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
        return ep_reward

    def _returns_advantages(self, rewards, dones, values, next_value):
        # next_value is the bootstrap value estimate of a future state (the critic)
        #returns = np.append(np.zeros_like(rewards), next_value[0:,]), I dont think this is right
        returns= np.append(values, np.squeeze(next_value))
        # returns are calculated as discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t+1] * (1-dones[t])
        returns = returns[:-1]
        # advantages are returns - baseline, value estimates in our case
        advantages = returns - values
        return returns, advantages
    #the return here is not actually reward. it has been modified. 
    def _value_loss(self, returns, pred_value):
        # value loss is typically MSE between value estimates and returns
        return self.params['value']*kls.mean_squared_error(returns, pred_value)

    def _all_losses(self,exp,pred_action):
        return exp 
    #this is actual log(pi(a|s)).not delta. action_v is mean
  
    def _compute_all_loss(self,pl,el,vl):
        #loss_policy=tf.reduce_mean(-log_prob_v)
        return pl+el+vl


























'''
test code

env = gym.make("Pong-v0")
observation = env.reset()
observation=observation/255.0
obs=np.empty((3,)+observation.shape)
obs[0]=observation
obs[1]=observation
obs[2]=observation
print(obs.shape)



model=Model()
#model.summary()
agent=A2CAgent(model)

action,sigma,value=agent.model.action_value(obs)
print(action)
'''
'''
env = gym.make('CartPole-v0')
model = Model(num_actions=env.action_space.n)
agent = A2CAgent(model)

rewards_sum = agent.test(env)
print("Total Episode Reward: %d out of 200" % agent.test(env))


logging.getLogger().setLevel(logging.INFO)

rewards_history = agent.train(env)
print("Finished training.")
'''