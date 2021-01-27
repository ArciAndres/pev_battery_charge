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
config.num_agents = 6
config.n_pevs = 20
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
for _ in range(env.total_timesteps-1):
    
    actions = [space.sample()*0.2 for space in env.action_space]
    
    #set_trace()
    obs, rewards, done, info, [] =  env.step(actions)
    env.render()
    
#%%
# Good manual gif generation

obs = env.reset()
for _ in range(env.total_timesteps-1):
    #sleep(0.01)
    actions = [space.sample()*0.2 for space in env.action_space]
    
    #set_trace()
    obs, rewards, done, info, [] =  env.step(actions)
    plug_pevs = env.cs_schedule[env.timestep]
    
    # Set zero to the other unplugged actions. 
    actions_last = [0 if -1 == plug_pevs[i] else a for i, a in enumerate(env.actions_last)]
    
    
    plt.rcParams['figure.figsize'] = [12, 5*2.8]
    plt.figure('BatteryCharge')

    # Charge bars of the charging stations    
    ind = [n for n in range(env.num_agents)]
    
    
    plt.subplot(4,1,1)
    p1 = plt.bar(ind, actions_last, width=0.35, )
    
    plt.ylim(0,1)
    
    xticks = []
    for i in range(env.num_agents):
        xtick = 'CS%d ' %i
        if plug_pevs[i] == -1:
            xtick += '(-1)'
        else:
            xtick += '(EV%d)' % plug_pevs[i]
        xticks.append(xtick)
    
    plt.xticks(ind, xticks)
    
    
    # Plot Gannt-like diagram
    plt.subplot(4,1,2)
    
    timesteps = [i for i in range(env.total_timesteps)]
    ind2 = [n for n in range(env.n_pevs)]
    EVnames = ['EV%d'%i for i in range(env.n_pevs)]

    for pev in env.pevs:
        plt.broken_barh([(pev.t_start, pev.t_end - pev.t_start)], (pev.id-0.25,0.5))
    
    plt.axvline(x=env.timestep, color='y')
    
    plt.yticks(ind2, EVnames)
    plt.ylim(-2,env.n_pevs+1)
    plt.xlim(0,env.total_timesteps)
    plt.grid(True)

    # Plot state of charge of the vehicles. 

    plt.subplot(4,1,3)
    
    ind2 = [n for n in range(env.n_pevs)]
    socs = [pev.soc for pev in env.pevs]
    soc_left = [pev.soc_ref - pev.soc for pev in env.pevs]
    
    p20 = plt.bar(ind2, socs, width=0.35, color='g' )
    p21 = plt.bar(ind2, soc_left, width=0.35, bottom=socs, color='pink')
    plt.plot()
    plt.xticks(ind2, EVnames)
    plt.ylim(0, 26) 
    
    plt.subplot(4,1,4)
    
    area_P = [(t, env.hist['area_P'][t]) for t in range(env.timestep)]
    
    plt.plot(timesteps, [env.area.P_ref for _ in timesteps])
    plt.plot(*zip(*area_P))
    plt.ylim(-2, env.area.P_ref*1.1 )
    plt.xlim(0,env.total_timesteps)
    plt.grid(True)
    
    plt.show()
    
# plt.legend((p1[0], p2[0]), ('Men', 'Women'))

# Make an analysis of high number of stations. 

#%%
obs = env.reset()
#for _ in range(env.total_timesteps-1):
#sleep(0.01)
actions = [space.sample()*0.2 for space in env.action_space]

#set_trace()
obs, rewards, done, info, [] =  env.step(actions)
env.render(plots=[1,2,3])
#%%
#set_trace()
env.reset()
env.plot_simulation(plots=[1,2,3])