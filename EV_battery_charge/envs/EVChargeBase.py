import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.random import random
from pev_battery_charge.utils.utils import createDict
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE


class PEV():
    def __init__(self, soc_max=24,
                       xi=0.1, 
                       p_min=0, 
                       p_max=22,
                       soc=0,
                       charge_time_desired=180,
                       target_charge=24
                       ):
        
        self.soc_max = soc_max
        self.xi = xi
        self.p_min = p_min
        self.p_max = p_max
        self.soc = soc
        self.charge_time_desired = charge_time_desired
        self.target_charge = target_charge
        
class ChargeStation():
    
    def __init__(self,n_pevs_max=50, 
                          n_pevs=0,
                          P_min=0,
                          P_max=35):
    
        self.n_pevs_max = n_pevs_max
        self.n_pevs = n_pevs
        self.P_min = P_min
        self.P_max = P_max

class EVChargeBase(MultiAgentEnv):
    
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
        
        self.distribute_load()

#----------------------------------------------------------------
#----------------- Distribution in load -------------------------
#----------------------------------------------------------------
                
    def distribute_load(self):
        
        np.random.seed(self.seed)
        
        # charge_samples = charge_duration_max/interval_length
        # total_samples_start = total_samples-charge_samples # Allowed start sample. Cannot start charging at very end of all, for example
        rate, proportional_dist = self.get_shrinking_rate()
        T_start = proportional_dist*self.total_samples
        
        for i , pev in enumerate(self.pevs):
            
            # Randomize the start time, from a point between it and the next random_start_coeff elements.
            # random_start_coeff is 1 normally
            if i == self.n_pevs-1:
                pev.t_start = np.floor(T_start[i]-(T_start[i]-T_start[i-self.random_start_coeff])*random())
                
            else:
                pev.t_start = np.floor(T_start[i]+(T_start[i+self.random_start_coeff]-T_start[i])*random())
                print(pev.t_start)
                
            charge_samples = pev.charge_time_desired/self.interval_length
            pev.t_end = np.floor(pev.t_start + charge_samples*(1-self.charge_duration_tolerance*(1-random())))
            
            # SOC can be any number between 0 and soc_max
            pev.soc = round(pev.soc_max*random()*self.initial_charge_max*100)/100
        
        self.update_df()
        
        
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
    
    def update_df(self):
        self.df = pd.DataFrame([pev.__dict__ for pev in self.pevs])
#-----------------------------------------------------------
#---------------- Compute greedy charge --------------------
#-----------------------------------------------------------
    
    def compute_straight_charge(self):
        '''
        
        Computes a greedy approach for charge, simply generating a straight line between
        the initial and target SOC value. The mean charge power is the slope of the line, m.
        The value of m should not exceed the p_max value. Otherwise, the charge goal not be
        reached in any method.
        
        '''
        for pev in self.pevs:
            pev.X = [t for t in range(int(pev.t_start), int(pev.t_end+1), 1)]
            
            # m is the slope of the straight line. It is a measure of power in [kW]
            pev.p_charge_rate = (pev.target_charge - pev.soc)/(pev.t_end - pev.t_start) / self.interval_length * 60
            b = pev.soc - pev.p_charge_rate * pev.t_start
            pev.Y = [pev.p_charge_rate*x_ + b for x_ in pev.X]
        
        self.update_df()
        
    def compute_pev_plugin(self):
        P, V = [], []

        for t in range(self.total_samples):
            p = 0
            n_plugged = 0
            for pev in self.pevs:
                if t >= pev.t_start and t <= pev.t_end:
                    n_plugged += 1
                    p += pev.p_charge_rate
            P.append(p)
            V.append(n_plugged)
            
        self.P_sim = P
        self.V_sim = V
            
        self.update_df()
        
    
#-----------------------------------------------------------
#------------------- Plotting methods ----------------------
#-----------------------------------------------------------
        
    def plot_common(self):
        # Add grid and limits the X axis in corresponding samples. 
        plt.grid(True)
        plt.xlim([0, self.total_samples]) 
        
    def plot_ax(self, plots, n_plots):
        n_plots += 1
        plt.subplot(len(plots),1,n_plots)
        
        self.plot_common()
        return n_plots
        
    def plot_simulation(self, plots=[1,2,3]):
        
        figlabel = "Simulation PEV Charge"
        if figlabel in plt.get_figlabels():    
            plt.close(figlabel)
        
        plt.rcParams['figure.figsize'] = [10, 5*len(plots)]
        plt.figure(figlabel)
        
        n_plots = 0
        samples = [i for i in range(self.total_samples)]
        
        if 1 in plots:
            assert hasattr(self.pevs[0],'X') # X must be created. Check in first sample
            n_plots = self.plot_ax(plots, n_plots)
            for pev in self.pevs:
                plt.plot(pev.X,pev.Y)
        
        # Consumed power across the total time
        if 2 in plots:
            assert hasattr(self,'P_sim') 
            n_plots = self.plot_ax(plots, n_plots)
            plt.step(samples, self.P_sim)
        
        # Vehicles connected at the same time
        if 3 in plots:
            assert hasattr(self,'V_sim') 
            n_plots = self.plot_ax(plots, n_plots)
            plt.step(samples, self.V_sim)
