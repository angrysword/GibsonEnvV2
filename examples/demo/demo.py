from gibson2.envs.locomotor_env import NavigateEnv, NavigateRandomEnv
from time import time
import numpy as np
from time import time
import gibson2
import os

if __name__ == "__main__":
    config_filename = os.path.join(os.path.dirname(gibson2.__file__),
                                   '../examples/configs/turtlebot_p2p_nav_house.yaml')
    nav_env = NavigateRandomEnv(config_file=config_filename, mode='gui')
    #nav_env = NavigateRandomEnv(config_file=config_filename )
    for j in range(3):
        nav_env.reset()
        for i in range(600):    # 300 steps, 30s world time
            s = time()
            action = nav_env.action_space.sample()
            action=np.array([float(action[0]),float(action[1])])
            ts = nav_env.step(action)
            print("step/reward: %s/%s"%(i,ts[1]))
            print(ts[1], 1 / (time() - s))
            if ts[2]:
                print("*************************************************")
                print("----------------Episode finished after {} timesteps".format(i + 1))
                break
    nav_env.clean()
