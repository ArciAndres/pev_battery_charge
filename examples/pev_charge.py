# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 19:46:43 2020

@author: andre
"""
#%matplotlib inline

import numpy as np
from pev_battery_charge.envs.PEVBatteryCharge import PEVBatteryCharge
from pdb import set_trace
from pev_battery_charge.envs.config_pev import get_config
from matplotlib import pyplot as plt
from time import sleep
config = get_config(notebook=True)

#set_trace()
env = PEVBatteryCharge(args=config)

# #%%
# env.build_random_schedule()
# env.compute_greedy_charge()
# env.compute_power_ideal()
# env.compute_pev_plugin()
# env.plot_simulation(plots=[1,3])

#%%

# from matplotlib import pyplot as plt

# for _ in range(10):
#     #sleep(1)
#     env.build_random_schedule()
#     env.compute_greedy_charge()
#     env.compute_power_ideal()
#     env.compute_pev_plugin()
#     env.plot_simulation(plots=[1,3])
#     plt.show()
#%%

actions = [space.sample() for space in env.action_space]

#set_trace()

obs, rewards, info, done, _ =  env.step(actions)
obs
#%%

obs = env.reset()
for _ in range(env.total_timesteps):
    #sleep(1)
    actions = [space.sample() for space in env.action_space]
    
    #set_trace()
    obs, rewards, done, info, [] =  env.step(actions)
    
    
    
    
width = 0.35
plt.bar(np.arange(env.num_agents), height, kwargs)
#%%
#set_trace()
env.reset()
env.plot_simulation(plots=[1,2,3])