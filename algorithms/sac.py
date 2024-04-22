import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from algorithms import models
from algorithms.multi_view_disentanglement import update_mvd, update_decoder

class SAC(object):
    def __init__(self, obs_shape, action_shape, action_range, cfg, priorioceptive_state_shape=None):
        self.cfg = cfg
        self.action_range = action_range
        self.device = cfg.device
        self.discount = cfg.discount
        self.critic_tau = cfg.critic_tau
        self.encoder_tau = cfg.encoder_tau
        self.actor_update_frequency = cfg.actor_update_freq
        self.critic_target_update_frequency = cfg.critic_target_update_freq
        self.batch_size = cfg.batch_size
        self.use_decoder = cfg.image_reconstruction_loss
        self.decoder_update_freq = cfg.decoder_update_freq
        self.mvd_update_freq = cfg.mvd_update_freq
        self.multi_view_disentanglement = cfg.multi_view_disentanglement
        self.num_cameras = len(cfg.cameras)

        self.actor = models.Actor(obs_shape, action_shape, cfg, priorioceptive_state_shape).to(self.device)

        self.critic = models.Critic(obs_shape, action_shape, cfg, priorioceptive_state_shape).to(self.device)
        self.critic_target = models.Critic(obs_shape, action_shape, cfg, priorioceptive_state_shape).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        if self.multi_view_disentanglement:
            all_enc_params = list(self.critic.shared_encoder.parameters()) + list(self.critic.private_encoder.parameters())
            self.mvd_optimizer = torch.optim.Adam(
                all_enc_params, lr=cfg.mvd_lr, betas=(cfg.mvd_beta, 0.999)
            )
        else:
            all_enc_params = self.critic.encoder.parameters()

        # tie conv layers between actor and critic
        if self.multi_view_disentanglement:
            self.actor.shared_encoder.copy_conv_weights_from(self.critic.shared_encoder)
            self.actor.shared_encoder.copy_head_weights_from(self.critic.shared_encoder)
            self.actor.private_encoder.copy_conv_weights_from(self.critic.private_encoder)
            self.actor.private_encoder.copy_head_weights_from(self.critic.private_encoder)
        else:
            self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(cfg.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_shape[0]

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr, betas=(0.5, 0.999))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        if self.use_decoder:
            self.decoder = models.Decoder((obs_shape[0] // self.num_cameras, *obs_shape[1:]), cfg).to(self.device)
            self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=cfg.mvd_lr, weight_decay=cfg.decoder_weight_lambda)
            self.encoder_optimizer = torch.optim.Adam(all_enc_params, lr=cfg.mvd_lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, proprioceptive_state=None, sample=False, eval_on_single_cam=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)

        if proprioceptive_state is not None:
            proprioceptive_state = torch.FloatTensor(proprioceptive_state).to(self.device).unsqueeze(0)

        dist, _ = self.actor(obs, proprioceptive_state, eval_on_single_cam=eval_on_single_cam)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, proprioceptive_state, action, reward, next_obs,
                      next_proprioceptive_state, not_done, logger, step):
        with torch.no_grad():
            dist, z = self.actor(next_obs, next_proprioceptive_state)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)

            target_Q1, target_Q2 = self.critic_target(next_obs, next_action, next_proprioceptive_state)

            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, proprioceptive_state)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, proprioceptive_state, logger, step):
        # detach conv filters, so we don't update them with the actor loss
        dist, z = self.actor(obs, proprioceptive_state, detach_encoder_conv=True, detach_encoder_head=self.multi_view_disentanglement)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        # detach conv filters, so we don't update them with the actor loss
        actor_Q1, actor_Q2 = self.critic(obs, action, proprioceptive_state, detach_encoder_conv=True, detach_encoder_head=self.multi_view_disentanglement)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()
        logger.log('train_alpha/loss', alpha_loss, step)
        logger.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step):
        (obs, action, reward, next_obs, not_done, proprioceptive_state, next_proprioceptive_state) = replay_buffer.sample(self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, proprioceptive_state, action, reward, next_obs, next_proprioceptive_state, not_done, logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, proprioceptive_state, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
            utils.soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)
            if self.multi_view_disentanglement:
                utils.soft_update_params(self.critic.shared_encoder, self.critic_target.shared_encoder,
                                         self.encoder_tau)
                utils.soft_update_params(self.critic.private_encoder, self.critic_target.private_encoder,
                                         self.encoder_tau)
            else:
                utils.soft_update_params(self.critic.encoder, self.critic_target.encoder, self.encoder_tau)

        if step % self.mvd_update_freq == 0:
            if self.cfg.multi_view_disentanglement:
                update_mvd(self, obs, next_obs, logger, step)

        if self.use_decoder:
            if step % self.decoder_update_freq == 0:
                update_decoder(self, obs, obs, logger, step)