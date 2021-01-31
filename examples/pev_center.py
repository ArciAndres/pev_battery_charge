# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 19:46:43 2020

@author: andre
"""
#%matplotlib inline

import numpy as np
from pev_battery_charge.envs.PEVBatteryChargeCentral import PEVBatteryChargeCentral
from pdb import set_trace
from pev_battery_charge.envs.config_pev import get_config
from matplotlib import pyplot as plt
from time import sleep
import pandas as pd
import imageio
config = get_config(notebook=True)
config.num_agents = 4
config.n_pevs = 6
#set_trace()

#%%
# To print the whole observation in console
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
#%%
env = PEVBatteryChargeCentral(args=config)

#%%


obs, _ = env.reset()
df = pd.DataFrame([obs])
print(df)


#%%
obs, _ = env.reset()
for i in range(env.total_timesteps):
    sleep(1)
    env.render()
    actions = env.action_space.sample()
    
    obs, rewards, done, info, _ = env.step(actions)
    df = pd.DataFrame([obs+[rewards]])
    print(df.to_string(header=False))
    
    
#%%
