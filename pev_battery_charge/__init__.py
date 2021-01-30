import gym

def register(id, entry_point, force=True):
    env_specs = gym.envs.registration.registry.env_specs
    if id in env_specs.keys():
        if not force:
            return
        del env_specs[id]
        gym.envs.registration.registry.register(id=id,entry_point=entry_point,)


register(
    id='pev-battery-charge-v0',
    entry_point='pev_battery_charge.envs:PEVBatteryCharge',
)

register(
    id='pev-battery-charge-central-v0',
    entry_point='pev_battery_charge.envs:PEVBatteryChargeCentral',
)




