import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.random import random
from pev_battery_charge.utils.utils import createDict
import gym
#from ray.rllib.env.multi_agent_env import MultiAgentEnv


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
        Time of the day (in TIMESTEP, NOT minutes) when the PEV is scheduled to be plugged in 
        The difference between t_start and t_end must match charge_time_desired.
    t_end : int
        Time of the day (in TIMESTEP, NOT minutes) when the PEV is scheduled to be plugged out
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
        
    def SOC_update(self, power_in, delta_t):
        
        self.soc += (1 - self.xi)*delta_t*power_in
        
        if self.soc > self.soc_max: # Cannot charge more if complete. 
                self.soc = self.soc_max

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
        Maximum power value (kW) capacity in the stations.
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
        self.p = 0 # Initial provided power is zero.
        self.plugged = plugged
        self.soc_left = 0
        self.t_left = 0
        self.pev_id = -1 # No PEV assigned 
        
class LoadArea():
    """ Load Area simulation. It handles the operations of the charging stations
    Parameters
    ----------
    P_max: float
        Maximum power value (kW) capacity in the area. 
    P_min: float
        Minimum power value (kW) capacity in the area.     
    P_ref : float
        Referece power value (kW) bound. The sum of the stations' power supply
        should not exceed this value. 
    
    """
    
    def __init__(self, 
                 P_max,
                 P_min,
                 P_ref,
                 charge_stations,
                 pevs):
        
        self.P_max = P_max
        self.P_min = P_min
        self.P_ref = P_ref
        self.charge_stations = charge_stations
        self.pevs = pevs
        self.P = 0        

class PEVChargeBase(gym.Env):
    
    """
    Core class of the EVCharge environment. It handles the interaction between
    PEVs, charge stations and the multiple variables.
    
    Parameters
    ----------
    
    area: 
    """
    
    def __init__(self, args):

        # This distribution of connections will put the agents across the total time, 
        # in an ordered pseudo random fashion
        
        self.pevs = self.area.pevs
        self.charge_stations = self.area.charge_stations
        self.sampling_time = args.sampling_time
        self.total_time = args.total_time
        self.charge_duration_tolerance = args.charge_duration_tolerance
        self.initial_charge_max = args.initial_charge_max
        self.random_start_coeff = args.random_start_coeff
        self.total_timesteps = int(self.total_time/self.sampling_time)
        
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()
        self.reset()
    
    
    def step(self, actions):
        """
        Apply power to the PEV, which will increase their SOC (if plugged-in)
        
        Parameters
        ----------
        actions : list of float
        
        """
        # Apply load to the cars PEV. Update SOC. 
        
        actions = self._preprocessAction(actions)
        self.actions_last = actions
        
        P = 0 # Total Power of the load area
        for cs, action in zip(self.charge_stations, actions):
            cs.p = action
            pev_id = self.cs_schedule[self.timestep][cs.id]
            if pev_id != -1:
                self.pevs[pev_id].SOC_update(cs.p, self.sampling_time)
            
            P += cs.p
        
        self.area.P = P
        
        # The variables are saved in the environment for debugging purposes. 
        self.obs = self._computeObservation()
        self.reward = self._computeReward()
        self.info = self._computeInfo()
        self.done = self._computeDone()
        
        self.update_history()
        self.schedule_step() # Plugs or unplugs vehicles depending on t
        self.timestep += 1
        
        return self.obs, self.reward, self.done, self.info, []
        
    def schedule_step(self):
        """ Synchronize the schedule with the values in the charging stations """
        
        for cs in self.charge_stations:
            cs.pev_id = self.cs_schedule[self.timestep][cs.id]
            cs.plugged = cs.pev_id != -1

    def update_history(self):
        self.hist['area_P'][self.timestep] = self.area.P
        
    def reset(self):
        self.timestep = 0
        self.build_random_schedule()
        self.compute_pev_plugin()
        self.schedule_step()
        self.obs = self._computeObservation()
        self.hist = {'area_P' : [0]*self.total_timesteps}
        
        
        return self.obs, []
    
    def set_seed(self, seed):
        self.seed = seed
        if self.seed is None:
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
        
        #np.random.seed(10)
        
        # charge_samples = charge_duration_max/sampling_time
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
                
            charge_samples = pev.charge_time_desired/self.sampling_time
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
            def st2h(st): # Sampling time to hour
                return st * self.sampling_time / 60
            
            pev.p_charge_rate = (pev.soc_ref - pev.soc)/(pev.t_end - pev.t_start)
            b = pev.soc - pev.p_charge_rate * pev.t_start
            pev.Y = [pev.p_charge_rate*x_ + b for x_ in pev.X]
        
        self.update_df()

#-----------------------------------------------------------
#---------------- Compute Plug-in Schedule -----------------
#-----------------------------------------------------------
    
    def compute_pev_plugin(self):
        """
        Creates the sequential schedule of plugged in vechiles according to the
        specified hour of connection and disconnection of the charging station.
        Doing so, at each timestep we know beforehand whether a ChargingStation 
        is occupied or not. 
        
        Computes
        -------
        plug_schedule : list
            len(plug_schedule) is the total_timesteps
            Each element of this list is a list containing the id of the plugged-in
            PEVs connected in a particular timestep
        cs_schedule : list
            Each element contains a list with num_agents elements indicating
            whether the stations are free (-1) or occuppied, for a particular
            timestep. It should correspond with plug_schedule.
        """
        plug_schedule = []
        
        for t in range(self.total_timesteps):
            plugs = [] # Current number of plugged-in PEVs in timestep
            for pev in self.pevs:
                if t >= pev.t_start and t <= pev.t_end:
                    plugs.append(pev.id)
            plug_schedule.append(plugs)
            
        self.plugged_sim = [len(p) for p in plug_schedule]
        self.plug_schedule = plug_schedule
        
        '''
        These two pieces of code are split so they are more readable, 
        and because plug_schedule is needed.
        ''' 
        
        # cs --> Charging Station
        cs_schedule = []
        pev_schedule = []

        # mappings from ChargingStations to PEV ids, and viceversa
        # -1 means
        cs2pev = [-1]*self.num_agents # -1 means that the station is free
        pev2cs = [-1]*self.n_pevs # -1 means that the PEV is not plugged in
        
        for t in range(self.total_timesteps):
            for pev_id in range(self.n_pevs): # 
                if pev_id in plug_schedule[t]: # Check if the PEV is plugged-in
                    if pev2cs[pev_id] == -1: # PEV should be in, but it is not
                        for i in range(len(cs2pev)): # Search a free station. 
                            if cs2pev[i] == -1: # Assign to the first free station
                                cs2pev[i] = pev_id
                                pev2cs[pev_id] = i
                                break # Stop searching for a free station
                else:
                    if pev2cs[pev_id] != -1: # If it was charging before, unplug it.
                        cs2pev[pev2cs[pev_id]] = -1
                        pev2cs[pev_id] = -1
                        
            cs_schedule.append(cs2pev.copy())
            pev_schedule.append(pev2cs.copy())
            
        self.cs_schedule = cs_schedule
        self.pev_schedule = pev_schedule
        
        return plug_schedule, cs_schedule, pev_schedule
    
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
            assert hasattr(self,'plugged_sim'), "Plugged-in simulation has not been performed."
            n_plots = self.plot_ax(plots, n_plots)
            plt.step(timesteps, self.plugged_sim)
