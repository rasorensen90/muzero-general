from gym.envs.registration import register

register(
    id='bhs-v0',
    entry_point='bhs.envs:BHSEnv',
)