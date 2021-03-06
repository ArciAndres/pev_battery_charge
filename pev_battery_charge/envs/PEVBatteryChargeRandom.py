import numpy as np
from pev_battery_charge.envs.PEVChargeCoreRandom import PEV, ChargeStation, PEVChargeBase, LoadArea
from gym import spaces

class PEVBatteryChargeRandom(PEVChargeBase):
    '''
    Plug-in Electric Vehicle Charing environment. 
    Centralized version of the actual problem. In this setting, it is assumed
    that a central controller in the load area performs the action calculation. 
    A simulation of a charging station as a multi-agent system.
    
    Parameters:
    ----------
    n_pevs: (int)
        Number of Plugged-in Electric Vehicles in the environment. 
    
    '''    
    
    def __init__(self, args):
        
        self.n_pevs = args.n_pevs
        self.num_agents = args.num_agents
        
        self.p_max = args.p_max
        self.p_min = args.p_min
        self.soc_ref = args.soc_ref
        self.charge_time_desired = args.charge_time_desired
        self.soc_initial = args.soc_initial
        self.soc_max = args.soc_max
        self.xi = args.xi
        self.P_max = args.P_max
        self.P_min = args.P_min
        self.P_ref = args.P_ref
        self.seed_ = args.seed
        self.rew_weights = [int(r) for r in args.reward_weights]
        self.actions_last = [0 for _ in range(self.num_agents)]
        self.action_weight = 10
        self.train_random = args.train_random
        
        if self.train_random:
            # So each station will have only 1 car, and that's it. No schedule.
            self.n_pevs = self.num_agents
        
        pevs = [PEV( ID=i,
                     soc_max=self.soc_max,
                     xi=self.xi,
                     soc=self.soc_initial, 
                     charge_time_desired=self.charge_time_desired) for i in range(self.n_pevs)]
        
        charge_stations = [ChargeStation(ID=i, 
                                  p_min=self.p_min, 
                                  p_max=self.p_max) for i in range(self.num_agents)]
        
        self.area = LoadArea(P_max=self.P_max, P_min=self.P_min, P_ref=self.P_ref, 
                             charge_stations=charge_stations, 
                             pevs=pevs)
        
        super().__init__(args=args)
        
    def _actionSpace(self):
        return spaces.Box(low=0, high=np.inf, shape=(self.num_agents,) , dtype=np.float32)
    
    def _observationSpace(self):
        shape = 4 + self.num_agents*2 # 4 Obs_general, soc_remain (num_agents), plugs (num_agents)
        
        return spaces.Box(low=0, high=np.inf, shape=(shape,), dtype=np.float32)
    
    def _preprocessAction(self, actions):
        # If the charging station is not connected, the applied action will be zero
        for i, cs in enumerate(self.charge_stations):
            if not cs.plugged:
               actions[i] = 0 
        
        return [action*self.action_weight for action in actions]
    
    def _computeReward(self):
        """ Reward has multiple weights and penalizations. """
        
        rewards = []
        self.info_rewards = []
        for cs in self.charge_stations:
            rew = [0]*len(self.rew_weights)
            if cs.plugged:
                
                if self.train_random:
                    # Agents and cars are correspondant. 
                    pev = self.pevs[cs.id]
                else:
                    pev = self.pevs[cs.pev_id]
                    
                # Penalization on remaining SOC
                soc_remain = -(pev.soc_ref - pev.soc)
                rew[0] = soc_remain/self.soc_ref # Normalized on the reference, not max
                
                # Penalization surpassing local limit
                if cs.p > cs.p_max or cs.p < cs.p_min:
                    rew[1] = (-1)
                
                # Penalization surpassing global limit
                if self.area.P > self.area.P_ref or self.area.P < self.area.P_min:
                    rew[2] = (-1)
            
            reward = np.array(rew)*self.rew_weights
            
            self.info_rewards.append(reward)
        
            rewards.append(sum(reward))
        
        self.info_rewards_sum = rewards
        rewards = sum(rewards) # A single joint reward value from all stations. 
        return rewards
    
    def _computeObservation(self):
        """
        Consider that the agents are the charging stations.
        Global information is collected, and a joint vector is returned.
        to the station is the total area power. 
        """
        
        socs_remain = []
        
        for cs in self.charge_stations:
            if cs.plugged:
                
                if self.train_random:
                    # Agents and cars are correspondant. 
                    pev = self.pevs[cs.id]
                else:
                    pev = self.pevs[cs.pev_id]
                    
                soc_remain = pev.soc_ref - pev.soc
            else:
                soc_remain = 0
            
            socs_remain.append(soc_remain)
        
        plugs = [cs.plugged for cs in self.charge_stations]
        #timesteps_remaining = pev.t_end - self.timestep
            
        # We assume the min and max power limit of each station is the same.    
        obs_general = [self.p_min, 
                       self.p_max,
                       self.area.P_ref,
                       self.area.P_ref - self.area.P]
        
        observation = obs_general + socs_remain + plugs
        
        return observation
        
    def _computeInfo(self):
        #raise NotImplementedError()
        return {"timestep: ": self.timestep, 
                "rewards_info": self.info_rewards, 
                "rewards_info_sum": self.info_rewards_sum}
    
    def _computeDone(self):
        #raise NotImplementedError()
        if self.timestep >= self.total_timesteps-1:
            return True
        else: 
            return False
        
         
        