# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 19:46:43 2020

@author: andre
"""
#%matplotlib inline

import numpy as np
from pev_battery_charge.envs.EVChargeEnv import EVChargeEnv
from pdb import set_trace

ev = EVChargeEnv(20, seed=1515)

#%%
ev.compute_straight_charge()
ev.compute_pev_plugin()
#%%


ev.plot_simulation(plots=[1])

#%%