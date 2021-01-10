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
    sleep(0.01)
    actions = [space.sample() for space in env.action_space]
    
    #set_trace()
    obs, rewards, done, info, [] =  env.step(actions)
    plug_pevs = env.cs_schedule[env.timestep]
    
    # Set zero to the other unplugged actions. 
    actions_last = [0 if -1 == plug_pevs[i] else a for i, a in enumerate(env.actions_last)]
    
    
    plt.rcParams['figure.figsize'] = [8, 5*2.5]
    plt.figure('BatteryCharge')

    # Charge bars of the charging stations    
    ind = [n for n in range(env.num_agents)]
    
    
    plt.subplot(3,1,1)
    p1 = plt.bar(ind, actions_last, width=0.35, )
    
    plt.ylim(0,3)
    plt.xticks(ind, ['CS%d (%d)'%(i, plug_pevs[i]) for i in range(env.num_agents)])
    
    
    # Plot Gannt-like diagram
    plt.subplot(3,1,2)
    pev = env.pevs[0]
    timesteps = [i for i in range(env.total_timesteps)]
    ind2 = [n for n in range(env.n_pevs)]
    EVnames = ['EV%d'%i for i in range(env.n_pevs)]
    
    barsX = [(pev.t_start, pev.t_end - pev.t_start) for pev in env.pevs]
    barsY = [(i,0.5) for i in range(env.n_pevs)]
    for pev in env.pevs:
        plt.broken_barh([(pev.t_start, pev.t_end - pev.t_start)], (pev.id-0.25,0.5))
    
    plt.axvline(x=env.timestep, color='y')
    
    plt.yticks(ind2, EVnames)
    plt.ylim(-2,env.n_pevs+1)
    plt.xlim(0,env.total_timesteps)
    plt.grid(True)

    # Plot state of charge of the vehicles. 

    plt.subplot(3,1,3)
    
    ind2 = [n for n in range(env.n_pevs)]
    socs = [pev.soc for pev in env.pevs]
    soc_left = [pev.soc_ref - pev.soc for pev in env.pevs]
    
    p20 = plt.bar(ind2, socs, width=0.35, color='g' )
    p21 = plt.bar(ind2, soc_left, width=0.35, bottom=socs, color='pink')
    plt.plot()
    plt.xticks(ind2, EVnames)
    plt.ylim(0, 26) 
    plt.show()
# plt.legend((p1[0], p2[0]), ('Men', 'Women'))

# Make an analysis of high number of stations. 


#%%
#set_trace()
env.reset()
env.plot_simulation(plots=[1,2,3])