# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 19:46:43 2020

@author: andre
"""
#%matplotlib inline

import numpy as np
from EV_battery_charge.envs.EVChargeEnv import EVChargeEnv
from pdb import set_trace

env = EVChargeEnv(n_pevs=20, n_stations=10, seed=1515)

#%%
env.compute_greedy_charge()
env.compute_power_ideal()
env.compute_pev_plugin()
#%%
env.plot_simulation(plots=[1,2,3])

#%%

plug_schedule = env.plug_schedule


#%%
    