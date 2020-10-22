import numpy as np
from pev_battery_charge.envs.EVChargeManagement import PEV, ChargeStation, ChargeManagement


class EVChargeEnv(ChargeManagement):
    
    def __init__(self, n_pevs, 
                       soc_max=24,
                       p_max=22,
                       p_min=0, 
                       target_charge=24,
                       charge_time_desired=180,
                       initial_soc=0,
                       xi=0.1,
                       P_max=31.5,
                       
                       interval_length=5,
                       total_time=960, 
                       charge_duration_tolerance=0.2,
                       initial_charge_max=0.5,
                       initial_charge_min=0,

                       seed=1515
                       ):
        
        self.n_pevs = n_pevs
        
        pevs = [PEV(soc_max=soc_max, xi=xi, p_min=p_min, p_max=p_max,
                         soc=initial_soc, charge_time_desired=charge_time_desired) for i in range(n_pevs)]
        
        station = ChargeStation(n_pevs=len(pevs), P_max=P_max)
        
        super().__init__(pevs=pevs, charge_station=station)
        
            
        
         
        