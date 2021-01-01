# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 19:46:43 2020

@author: andre
"""
#%matplotlib inline

import numpy as np
from EV_battery_charge.envs.EVChargeEnv import EVChargeEnv
from pdb import set_trace
from config_ev1 import get_config 

config = get_config(notebook=True)
#%%
env = EVChargeEnv(args=config)

#%%
env.build_random_schedule()
env.compute_greedy_charge()
env.compute_power_ideal()
env.compute_pev_plugin()
env.plot_simulation(plots=[1,2,3])

#%%

actions = [space.sample() for space in env.action_space]

#%%
#set_trace()
env.reset()
env.plot_simulation(plots=[1,2,3])