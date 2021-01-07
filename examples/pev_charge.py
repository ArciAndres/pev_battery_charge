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

config = get_config(notebook=True)
config.num_agents = 4
#%%

config.n_pevs = 10

env = PEVBatteryCharge(args=config)

#%%
env.build_random_schedule()
env.compute_greedy_charge()
env.compute_power_ideal()
env.compute_pev_plugin()
env.plot_simulation(plots=[1,3])

#%%

from time import sleep
from matplotlib import pyplot as plt

for _ in range(10):
    #sleep(1)
    env.build_random_schedule()
    env.compute_greedy_charge()
    env.compute_power_ideal()
    env.compute_pev_plugin()
    env.plot_simulation(plots=[1,3])
    plt.show()
#%%

actions = [space.sample() for space in env.action_space]

#set_trace()

obs, rewards, info, done =  env.step(actions)

#%%
#set_trace()
env.reset()
env.plot_simulation(plots=[1,2,3])