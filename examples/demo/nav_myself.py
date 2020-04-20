from gibson2.envs.locomotor_env import NavigateEnv, NavigateRandomEnv
from time import time
import numpy as np
from time import time
import gibson2
import os

if __name__ == "__main__":
    config_filename = os.path.join(os.path.dirname(gibson2.__file__),
                                   '../examples/configs/turtlebot_p2p_nav_house.yaml')
    nav_env = NavigateEnv(config_file=config_filename,
      mode='gui',
      action_timestep=1.0 / 10.0,
      physics_timestep=1.0 / 40.0)

    for episode in range(10):
        print('Episode: {}'.format(episode))
        nav_env.reset()
        for step in range(500):  # 500 steps, 50s world time
            action = nav_env.action_space.sample()
            print(action)
            state, reward, done, _ = nav_env.step(action)   
       
