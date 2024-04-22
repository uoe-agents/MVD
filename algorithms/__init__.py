from algorithms.sac import SAC
from algorithms.drq import DrQ

algorithm = {
	'sac': SAC,
	'drq': DrQ,
}

def make_agent(obs_shape, action_shape, action_range, cfg, proprioceptive_state_shape=None):
	return algorithm[cfg.algorithm](obs_shape, action_shape, action_range, cfg, proprioceptive_state_shape)