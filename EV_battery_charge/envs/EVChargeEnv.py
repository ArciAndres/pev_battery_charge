import numpy as np
from EV_battery_charge.envs.EVChargeCore import PEV, ChargeStation, EVChargeBase, LoadArea
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
                       P_min=0,
                       interval_length=5,
                       total_time=960,
                       initial_charge_max=0.5,
                       initial_charge_min=0,
                       seed=1515,
                       charge_duration_tolerance=0.2,
                       random_start_coeff=1,
                       pevs=None,
                       ):
        
        self.n_pevs = n_pevs
        self.n_stations = n_stations
        
        pevs = [PEV( ID=i,
                     soc_max=soc_max, 
                     xi=xi,
                     soc=initial_soc, 
                     charge_time_desired=charge_time_desired) for i in range(n_pevs)]
        
        charge_stations = [ChargeStation(ID=i, 
                                  p_min=p_min, 
                                  p_max=p_max) for i in range(n_stations)]
        
        load_area = LoadArea(P_max=P_max, P_min=P_min, 
                             charge_stations=charge_stations, 
                             pevs=pevs)
        
        super().__init__(area=load_area)
        
    def _actionSpace(self):
        return [ spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32) for _ in self.n_stations]
    
    def _observationSpace(self):
        return [ spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32) for _ in self.n_stations]
    
    def _preprocessAction(self, actions):
        # Here we could clip 
        return actions
    
    def _computeReward(self):
        """ Reward has multiple weights and penalizations. """
        
        rewards = []
        
        
        
        for cs in self.charge_stations:
            rew = [0]*5
            if cs.plugged:
                pev = self.pevs[cs.pev_id]
                
                # Penalization on remaining SOC
                soc_remain = pev.soc - pev.soc_ref
                rew[0] = soc_remain/self.soc_ref # Normalized on the reference, not max
                
                # Penalization surpassing local limit
                if cs.p > cs.p_max or cs.p < cs.p_min:
                    rew[1] = (-1)
                
                # Penalization surpassing global limit
                if self.area.P > self.area.P_max or self.area.P < self.area.P_min:
                    rew[2] = (-1)
            
            reward = np.array(rew)*self.rew_weights
            
            rewards.append(sum(reward))                
        
    def _computeObservation(self):
        """
        Consider that the agents are the charging stations.
        Only local information is collected. The only global information known
        to the station is the total area power. 
        
        """
        
        observations = []
                
        for cs in self.charge_stations:
            pev = self.pevs[cs.pev_id]
            soc_remain = pev.soc - pev.soc_ref
            timesteps_remaining = pev.t_end - self.timestep
            observations.append([cs.p_min, 
                                 cs.p_max,
                                 self.area.P_max,
                                 cs.plugged, 
                                 soc_remain,
                                 timesteps_remaining, 
                                 self.area.P])
        
        return observations
        
        
    def _computeInfo(self):
        raise NotImplementedError()
    
    def _computeDone(self):
        raise NotImplementedError()
        
         
        