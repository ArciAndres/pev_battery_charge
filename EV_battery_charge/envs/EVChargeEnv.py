import numpy as np
from EV_battery_charge.envs.EVChargeCore import PEV, ChargeStation, EVChargeBase
from gym import spaces

class EVChargeEnv(EVChargeBase):
    '''
    Plug-in Electric Vehicle Charing environment. 
    A simulation of a charging station as a multi-agent system.
    
    Parameters:
    ----------
    
    '''    
    
    def __init__(self, n_pevs, 
                       n_stations,
                       soc_max=24,
                       p_max=22,
                       p_min=0, 
                       soc_ref=24,
                       charge_time_desired=180,
                       initial_soc=0,
                       xi=0.1,
                       P_max=200,
                       interval_length=5,
                       total_time=960,
                       initial_charge_max=0.5,
                       initial_charge_min=0,
                       seed=1515,
                       charge_duration_tolerance=0.2,
                       pevs=None
                       ):
        
        self.n_pevs = n_pevs
        self.n_stations = n_stations
        
        pevs = [PEV(soc_max=soc_max, xi=xi, p_min=p_min, p_max=p_max,
                         soc=initial_soc, charge_time_desired=charge_time_desired) for i in range(n_pevs)]
        
        station = [ChargeStation(n_pevs=len(pevs), P_max=P_max) for _ in n_stations]
        
        super().__init__(pevs=pevs, charge_station=station)
        
    def _actionSpace(self):
        
        '''
        Return the action space of the environment, a Dict of Box(4,) with NUM_DRONES entries 
        '''
        
        #### Action vector ######## P0            P1            P2            P3
        act_lower_bound = np.array([-1,           -1,           -1,           -1])
        act_upper_bound = np.array([1,            1,            1,            1])
        
        return spaces.Dict({ str(i): spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32) for i in range(self.NUM_DRONES) })

    ####################################################################################################
    #### Return the observation space of the environment, a Dict with NUM_DRONES entries of Dict of ####
    #### { Box(4,), MultiBinary(NUM_DRONES) } ##########################################################
    ####################################################################################################
    def _observationSpace(self):
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WR       WP       WY       P0            P1            P2            P3
        obs_lower_bound = np.array([-1,      -1,      0,      -1,  -1,  -1,  -1,  -1,     -1,     -1,     -1,      -1,      -1,      -1,      -1,      -1,      -1,           -1,           -1,           -1])
        obs_upper_bound = np.array([1,       1,       1,      1,   1,   1,   1,   1,      1,      1,      1,       1,       1,       1,       1,       1,       1,            1,            1,            1])
        return spaces.Dict({ str(i): spaces.Dict ({"state": spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32),
                                                    "neighbors": spaces.MultiBinary(self.NUM_DRONES) }) for i in range(self.NUM_DRONES) })

    def _computeReward(self):
        raise NotImplementedError()
        
    def _computeObservation(self):
        raise NotImplementedError()
        
    def _computeInfo(self):
        raise NotImplementedError()
    
    def _computeDone(self):
        raise NotImplementedError()
        
         
        