import numpy as np
from pev_battery_charge.envs.PEVChargeCore import PEV, ChargeStation, PEVChargeBase, LoadArea
from gym import spaces

class PEVBatteryCharge(PEVChargeBase):
    '''
    Plug-in Electric Vehicle Charing environment. 
    A simulation of a charging station as a multi-agent system.
    
    Parameters:
    ----------
    
    '''    
    
    def __init__(self, args):
        
        self.n_pevs = args.n_pevs
        self.n_stations = args.num_agents
        
        self.p_max = args.p_max
        self.p_min = args.p_min
        self.soc_ref = args.soc_ref
        self.charge_time_desired = args.charge_time_desired
        self.soc_initial = args.soc_initial
        self.soc_max = args.soc_max
        self.xi = args.xi
        self.P_max = args.P_max
        self.P_min = args.P_min
        self.seed = args.seed
        self.rew_weights = args.reward_weights
                       
        pevs = [PEV( ID=i,
                     soc_max=self.soc_max,
                     xi=self.xi,
                     soc=self.soc_initial, 
                     charge_time_desired=self.charge_time_desired) for i in range(self.n_pevs)]
        
        charge_stations = [ChargeStation(ID=i, 
                                  p_min=self.p_min, 
                                  p_max=self.p_max) for i in range(self.n_stations)]
        
        self.area = LoadArea(P_max=self.P_max, P_min=self.P_min, 
                             charge_stations=charge_stations, 
                             pevs=pevs)
        
        super().__init__(args=args)
        
    def _actionSpace(self):
        return [ spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32) for _ in range(self.n_stations)]
    
    def _observationSpace(self):
        return [ spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32) for _ in range(self.n_stations)]
    
    def _preprocessAction(self, actions):
        # Here we could clip 
        return [action[0] for action in actions]
    
    def _computeReward(self):
        """ Reward has multiple weights and penalizations. """
        
        rewards = []
        
        for cs in self.charge_stations:
            rew = [0]*len(self.rew_weights)
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
        
        return rewards
    
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
        #raise NotImplementedError()
        return {"timestep: ", self.timestep, ""}
    
    def _computeDone(self):
        #raise NotImplementedError()
        return -1
        
         
        