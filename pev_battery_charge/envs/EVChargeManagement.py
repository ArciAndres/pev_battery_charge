import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from random import random
from ..utils.utils import createDict


class PEV():
    def __init__(self, soc_max=24,
                       xi=0.1, 
                       p_min=0, 
                       p_max=22,
                       soc=0,
                       time_in=0, 
                       time_out=180
                       ):
        
        self.soc_max = soc_max
        self.xi = xi
        self.p_min = p_min
        self.p_max = p_max
        self.soc = soc
        self.time_in = time_in
        self.time_out = time_out
        
class ChargeStation():
    
    def __init__(self,n_pevs_max=20, 
                          n_pevs=0,
                          P_min=0,
                          P_max=35):
    
        self.n_pevs_max = n_pevs_max
        self.n_pevs = n_pevs
        self.P_min = P_min
        self.P_max = P_max

class BatteryCharge():
    
    def __init__(self, n_agents=20, 
                       interval_length=5,
                       capacity=24,
                       p_max=22,
                       p_min=0, 
                       target_charge=24,
                       total_time=960, 
                       charge_duration_max=180,
                       charge_duration_tolerance=0.2,
                       initial_charge_max=0.5,
                       P_max=20,
                       seed=1515
                 ):
        
        # This distribution of connections will put the agents across the total time, in an ordered pseudo random fashion
        
        self.n_agents = n_agents
        self.interval_length = interval_length
        self.capacity = [capacity]*n_agents
        self.p_max = [p_max]*n_agents
        self.p_min = [p_min]*n_agents
        self.target_charge = [target_charge]*n_agents
        self.total_time = total_time
        self.charge_duration_max = charge_duration_max
        self.charge_duration_tolerance = charge_duration_tolerance 
        self.initial_charge_max = initial_charge_max
        self.P_max = P_max
        
        total_samples = int(total_time/interval_length)
        charge_samples = charge_duration_max/interval_length
        
        total_samples_start = total_samples-charge_samples # Allowed start sample. Cannot start charging at very end of all, for example
        separation = total_samples_start/n_agents
        
        starts, ends, socs = [], [], []
        
        for i in range(n_agents):
            
            start_time = np.floor(separation*(i+random()))
            end_time = np.floor(start_time + charge_samples*(1-charge_duration_tolerance*(1-random())))
            
            #SOC can be any number between 0 and capacity
            socs.append(round(self.capacity[i]*random()*max_initial_charge*100)/100)
            starts.append(start_time)
            ends.append(end_time)
        
        
        
    
        
        

    
        
        
        
        
        