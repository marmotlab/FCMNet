from gym.envs.registration import register

register(
    id='MultiagentCatch-v0',
    entry_point='environments.scenarios:MultiAgentEnv'
)
