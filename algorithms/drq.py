import torch
import torch.nn.functional as F
import utils
from algorithms.sac import SAC
from algorithms.multi_view_disentanglement import update_mvd

class DrQ(SAC):
	def __init__(self, obs_shape, action_shape, action_range, cfg, proprioceptive_state_shape):
		super().__init__(obs_shape, action_shape, action_range, cfg, proprioceptive_state_shape)

		assert not self.use_decoder

	def update_critic(self, obs, proprioceptive_state, action, reward, next_obs, next_proprioceptive_state, not_done, obs_aug, next_obs_aug, logger, step):
		with torch.no_grad():
			dist, _ = self.actor(next_obs, next_proprioceptive_state)
			next_action = dist.rsample()
			log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
			target_Q1, target_Q2 = self.critic_target(next_obs, next_action, next_proprioceptive_state)
			target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
			target_Q = reward + (not_done * self.discount * target_V)

			dist_aug, _ = self.actor(next_obs_aug, next_proprioceptive_state)
			next_action_aug = dist_aug.rsample()
			log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1, keepdim=True)
			target_Q1, target_Q2 = self.critic_target(next_obs_aug, next_action_aug, next_proprioceptive_state)
			target_V = torch.min(
				target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug
			target_Q_aug = reward + (not_done * self.discount * target_V)

			target_Q = (target_Q + target_Q_aug) / 2

		# get current Q estimates
		current_Q1, current_Q2 = self.critic(obs, action, proprioceptive_state)
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		Q1_aug, Q2_aug = self.critic(obs_aug, action, proprioceptive_state)
		critic_loss += F.mse_loss(Q1_aug, target_Q) + F.mse_loss(Q2_aug, target_Q)

		logger.log('train_critic/loss', critic_loss, step)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		self.critic.log(logger, step)

	def update(self, replay_buffer, logger, step):
		(obs, action, reward, next_obs, not_done, obs_aug, next_obs_aug,
		 proprioceptive_state, next_proprioceptive_state) = replay_buffer.sample_drq(self.batch_size)

		self.aug_trans = replay_buffer.aug_trans

		logger.log('train/batch_reward', reward.mean(), step)

		self.update_critic(obs, proprioceptive_state, action, reward, next_obs, next_proprioceptive_state, not_done, obs_aug, next_obs_aug, logger, step)

		if step % self.actor_update_frequency == 0:
			self.update_actor_and_alpha(obs, proprioceptive_state, logger, step)

		if step % self.critic_target_update_frequency == 0:
			utils.soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
			utils.soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)
			if self.multi_view_disentanglement:
				utils.soft_update_params(self.critic.shared_encoder, self.critic_target.shared_encoder, self.encoder_tau)
				utils.soft_update_params(self.critic.private_encoder, self.critic_target.private_encoder, self.encoder_tau)
			else:
				utils.soft_update_params(self.critic.encoder, self.critic_target.encoder, self.encoder_tau)

		if step % self.mvd_update_freq == 0:
			if self.cfg.multi_view_disentanglement:
				update_mvd(self, obs, next_obs, logger, step)