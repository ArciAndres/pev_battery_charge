import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.random import random
from EV_battery_charge.utils.utils import createDict
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class PEV():
    '''
    Plug-in Electric Vehicle class.
    It simulates the vehicle that gets connected to the station to charge itself.
    
    Parameters
    ----------
    soc_max : int 
        Complete level of SOC (State of charge)
    xi : float
        Conversion losses parameter
    soc : float
        State Of Charge. Energy stored in the PEV battery currently. 
        Zero means totally discharged. 
    charge_time_desired : int
        Defined in minutes. Expected time of charge when plugged in to the station
    soc_ref : float
        Expected level of SOC to be reached during charge.
    t_start : int
        Time of the day (in minutes) when the PEV is scheduled to be plugged in 
        The difference between t_start and t_end must match charge_time_desired.
    t_end : int
        Time of the day (in minutes) when the PEV is scheduled to be plugged out
        The difference between t_start and t_end must match charge_time_desired.
    '''
    def __init__(self, ID, 
                       soc_max=24,
                       xi=0.1, 
                       soc=0,
                       charge_time_desired=180,
                       soc_ref=24,
                       t_start=None,
                       t_end=None
                       ):
        
        self.id = ID
        self.soc_max = soc_max
        self.xi = xi
        self.soc = soc
        self.charge_time_desired = charge_time_desired
        self.soc_ref = soc_ref
        self.t_start = t_start
        self.t_end = t_end
        
class ChargeStation():
    ''' 
    Charge station class. 
    (Virtual Charge Stations. They are not necessarily the same in a real world
     setup) In this setup, this element is initialized as a configuration object,
    and it is assumed that every PEV that plugs-in to the load area uses a free 
    charging station with these parameters.
    
    
    
    Parameters
    ----------
    p_min : float
        Minimum power value (kW) to be delivered PEV.
    p_max : float
        Maximum power value (kW) to be delivered to the PEV.
    plugged : bool
        Indicates if the charge station has a PEV plugged-in or it is free. 
    soc_left : float
        Only when plugged. Amount of SOC remaining to be delivered to the PEV.
    t_left : int
        Only when plugged. Remaining time (minutes) for the PEV to plug out. 
    '''
    
    def __init__(self, ID,
                       p_min=0,
                       p_max=22,
                       plugged=False):
        
        self.id = ID
        self.p_min = p_min
        self.p_max = p_max
        self.plugged = plugged
        self.soc_left = 0
        self.t_left = 0
        
class LoadArea():
    def __init__(self, 
                 P_max,
                 P_min,
                 charge_stations,
                 pevs):
        
        self.P_max = P_max
        self.P_min = P_min
        self.stations = charge_stations
        self.pevs = pevs
        
        self.plug_map = { station.ID: -1 for station in self.stations }

class EVChargeBase(MultiAgentEnv):
    
    """
    Core class of the EVCharge environment. It handles the interaction between
    PEVs, charge stations and the multiple variables.
    
    Parameters
    ----------
    
    pevs: 
    """
    
    def __init__(self, area,
                       interval_length=5,
                       total_time=960, 
                       charge_duration_tolerance=0.2,
                       initial_charge_max=0.5,
                       initial_charge_min=0,
                       random_start_coeff=1,
                       seed=1515,
                 ):

        # This distribution of connections will put the agents across the total time, 
        # in an ordered pseudo random fashion
        
        self.n_pevs = len(pevs)
        self.pevs = area.pevs
        self.charge_stations = charge_stations
        self.interval_length = interval_length
        self.total_time = total_time
        self.charge_duration_tolerance = charge_duration_tolerance 
        self.initial_charge_max = initial_charge_max
        self.initial_charge_min = initial_charge_min
        self.random_start_coeff = random_start_coeff
        self.seed = seed
        
        self.total_timesteps = int(total_time/interval_length)
        
        self.build_random_schedule()
    
    
    def step(self, action_dict):
        
        # Apply load to the cars PEV. Update SOC. 
        self.timestep += 1
                
        self._computeReward()
        self._computeObservation()
        self._computeInfo()
        self._computeDone()
        
        self.update_charge_stations() # Plugs or unplugs vehicles depending on t
    
    def reset(self):
        self.timestep = 0
        self.build_random_schedule()
        self.update_charge_stations()

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)
        
        
#----------------------------------------------------------------
#----------------- Distribution in load -------------------------
#----------------------------------------------------------------
                
    def build_random_schedule(self):
        """
        Performs a distribution of the load between the based on the parameters
        of the given PEVs, scheduling an pseudo-random hour of charge and initial
        SOC over the total programmed charge time.
        It is used as a reset to the environment. It is not needed when the 
        schedule is provided.         
        """
        
        np.random.seed(self.seed)
        
        # charge_samples = charge_duration_max/interval_length
        # total_timesteps_start = total_timesteps-charge_samples # Allowed start sample. 
        # Cannot start charging at very end of all, for example
        rate, proportional_dist = self.get_shrinking_rate()
        T_start = proportional_dist*self.total_timesteps
        
        for i , pev in enumerate(self.pevs):
            
            # Randomize the start time, from a point between it and the next random_start_coeff elements.
            # random_start_coeff is 1 normally
            if i == self.n_pevs-1:
                pev.t_start = np.floor(T_start[i]-(T_start[i]-T_start[i-self.random_start_coeff])*random())
                
            else:
                pev.t_start = np.floor(T_start[i]+(T_start[i+self.random_start_coeff]-T_start[i])*random())
                #print(pev.t_start)
                
            charge_samples = pev.charge_time_desired/self.interval_length
            pev.t_end = np.floor(pev.t_start + charge_samples*(1-self.charge_duration_tolerance*(1-random())))
            
            # SOC can be any number between 0 and soc_max
            pev.soc = round(pev.soc_max*random()*self.initial_charge_max*100)/100
        
        self.update_df()
        
        
    def get_proportional_initial_dist(self):
        '''
        Distributes the PEVs along the total charge time proportionally to their own
        charge time. 
        
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
    
    def compute_greedy_charge(self):
        '''
        
        Computes a greedy approach for charge, simply generating a straight line between
        the initial and target SOC value. The mean charge power is the slope of the line, m.
        The value of m should not exceed the p_max value. Otherwise, the charge goal 
        will not be reached by any method.
        
        '''
        for pev in self.pevs:
            pev.X = [t for t in range(int(pev.t_start), int(pev.t_end+1), 1)]
            
            # m is the slope of the straight line. It is a measure of power in [kW]
            pev.p_charge_rate = (pev.soc_ref - pev.soc)/(pev.t_end - pev.t_start) / self.interval_length * 60
            b = pev.soc - pev.p_charge_rate * pev.t_start
            pev.Y = [pev.p_charge_rate*x_ + b for x_ in pev.X]
        
        self.update_df()
    
    def compute_pev_plugin(self):
        pluggled_sim = []

        for t in range(self.total_timesteps):
            n_plugged = 0
            for pev in self.pevs:
                if t >= pev.t_start and t <= pev.t_end:
                    n_plugged += 1
            pluggled_sim.append(n_plugged)
            
        self.pluggled_sim = pluggled_sim
        self.update_df()
    
    def compute_power_ideal(self):

        assert hasattr(self.pevs[0], 'p_charge_rate'), \
            "compute_greedy_charge method must be executed before."
             
        P = []

        for t in range(self.total_timesteps):
            p = 0
            for pev in self.pevs:
                if t >= pev.t_start and t <= pev.t_end:
                    p += pev.p_charge_rate
            P.append(p)
        self.P_sim = P
                
#-----------------------------------------------------------
#------------------- Plotting methods ----------------------
#-----------------------------------------------------------
        
    def plot_common(self):
        # Add grid and limits the X axis in corresponding samples. 
        plt.grid(True)
        plt.xlim([0, self.total_timesteps]) 
        
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
        timesteps = [i for i in range(self.total_timesteps)]
        
        if 1 in plots:
            assert hasattr(self.pevs[0],'X') # X must be created. Check in first sample
            n_plots = self.plot_ax(plots, n_plots)
            for pev in self.pevs:
                plt.plot(pev.X,pev.Y)
        
        # Consumed power across the total time
        if 2 in plots:
            assert hasattr(self,'P_sim'), "Power simulation has not been performed."
            n_plots = self.plot_ax(plots, n_plots)
            plt.step(timesteps, self.P_sim)
        
        # Vehicles connected at the same time
        if 3 in plots:
            assert hasattr(self,'pluggled_sim'), "Plugged-in simulation has not been performed."
            n_plots = self.plot_ax(plots, n_plots)
            plt.step(timesteps, self.pluggled_sim)
