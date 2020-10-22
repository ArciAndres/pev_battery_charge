import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from random import random
from pev_battery_charge.utils.utils import createDict
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE


class PEV():
    def __init__(self, soc_max=24,
                       xi=0.1, 
                       p_min=0, 
                       p_max=22,
                       soc=0,
                       charge_time_desired=180
                       ):
        
        self.soc_max = soc_max
        self.xi = xi
        self.p_min = p_min
        self.p_max = p_max
        self.soc = soc
        self.charge_time_desired = charge_time_desired
        
class ChargeStation():
    
    def __init__(self,n_pevs_max=50, 
                          n_pevs=0,
                          P_min=0,
                          P_max=35):
    
        self.n_pevs_max = n_pevs_max
        self.n_pevs = n_pevs
        self.P_min = P_min
        self.P_max = P_max

class ChargeManagement(MultiAgentEnv):
    
    def __init__(self, pevs, 
                       charge_station,
                       
                       interval_length=5,
                       total_time=960, 
                       charge_duration_tolerance=0.2,
                       initial_charge_max=0.5,
                       initial_charge_min=0,
                       random_start_coeff = 1,
                       seed=1515,
                       

                 ):
        
        # This distribution of connections will put the agents across the total time, in an ordered pseudo random fashion
        
        self.n_pevs = len(pevs)
        self.pevs = pevs
        self.charge_station = charge_station
        self.interval_length = interval_length
        self.total_time = total_time
        self.charge_duration_tolerance = charge_duration_tolerance 
        self.initial_charge_max = initial_charge_max
        self.initial_charge_min = initial_charge_min
        self.random_start_coeff = random_start_coeff
        self.seed = seed
        
        self.total_samples = int(total_time/interval_length)
        
        

                
    def distribute_load(self):
        
        # charge_samples = charge_duration_max/interval_length
        # total_samples_start = total_samples-charge_samples # Allowed start sample. Cannot start charging at very end of all, for example
        rate, proportional_dist = self.get_shrinking_rate()
        T_start = proportional_dist*self.total_samples
        
        separation = 1
        for i , pev in enumerate(self.pevs):
            
            # Randomize the start time, from a point between it and the next random_start_coeff elements.
            # random_start_coeff is 1 normally
            if i == self.n_pevs-1:
                pev.t_start = np.floor(T_start[i]-(T_start[i]-T_start[i-self.random_start_coeff])*random())
                
            else:
                pev.t_start = np.floor(T_start[i]+(T_start[i+self.random_start_coeff]-T_start[i])*random())
            
            charge_samples = pev.charge_time_desired/self.interval_length
            pev.t_end = np.floor(pev.t_start + charge_samples*(1-self.charge_duration_tolerance*(1-random())))
            
            # SOC can be any number between 0 and soc_max
            pev.soc = round(pev.soc_max*random()*self.initial_charge_max*100)/100
        
        
        
    def get_proportional_initial_dist(self):
        '''
        Distributes the PEVs along the total charge time proportionally to their own
        charge time. 
        
        Example:
        '''
        
        total_charge_time_no_overlap = sum([pev.charge_time_desired for pev in self.pevs])
        
        proportional_dist = [pev.charge_time_desired/total_charge_time_no_overlap for pev in self.pevs]        
        return np.add.accumulate(proportional_dist)-proportional_dist[0]
        
    def get_shrinking_rate(self):
        '''
        Shrinkes the proportional distribution so that all desired end times of desired
        charge are inside the Total charge time.
        '''
        dist = self.get_proportional_initial_dist()
        
        end_times = []
        for start, pev in zip(dist, self.pevs):
            end_time = start*self.total_time + pev.charge_time_desired
            end_times.append(end_time)
            
        if max(end_times) > self.total_time:            
            shrink_rate = self.total_time/max(end_times)
            
        else:
            shrink_rate = 1 # no need to shrink
        
        return shrink_rate, dist*shrink_rate
    
   # def plot