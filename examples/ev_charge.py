# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 19:46:43 2020

@author: andre
"""

from pev_battery_charge.envs.EVChargeEnv import EVChargeEnv

ev = EVChargeEnv(20)

#%%
ev.compute_straight_charge()

#%%
ev.plot_simulation(subplot=True, plots=[1,2,3])