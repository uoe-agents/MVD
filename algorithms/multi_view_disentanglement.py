import torch
import torch.nn.functional as F
import numpy as np
import utils
from algorithms.info_nce import InfoNCE

def update_decoder(agent, obs, target_obs, logger, step):
    z, z_shared, z_private = agent.critic.get_representation(obs)
    batch_size = z.shape[0]
    num_cameras = z.shape[1]

    if agent.multi_view_disentanglement:
        shared_idx = np.random.randint(agent.num_cameras)
        z_shared = z_shared[:, shared_idx].unsqueeze(1).repeat(1, agent.num_cameras, 1)
        z = torch.concat((z_shared, z_private), dim=-1)
        z = z.reshape((batch_size*num_cameras, -1))

    target_obs = target_obs.reshape((batch_size*num_cameras, target_obs.shape[1] // num_cameras, *target_obs.shape[2:]))

    if target_obs.dim() == 4:
        # preprocess images to be in [-0.5, 0.5] range
        target_obs = utils.preprocess_obs(target_obs)
    rec_obs = agent.decoder(z)
    rec_loss = F.mse_loss(target_obs, rec_obs)

    logger.log('train_encoder/recon_loss', rec_loss, step)

    agent.encoder_optimizer.zero_grad()
    agent.decoder_optimizer.zero_grad()
    rec_loss.backward()
    agent.encoder_optimizer.step()
    agent.decoder_optimizer.step()

    agent.decoder.log(logger, step)

def update_mvd(agent, obs, next_obs, logger, step):
    z, z_shared, z_private = agent.critic.get_representation(obs)
    num_cameras = z.shape[1]

    with torch.no_grad():
        z_target, z_target_shared, z_target_private = agent.critic_target.get_representation(obs)
        next_z_target, next_z_target_shared, next_z_target_private = agent.critic_target.get_representation(next_obs)

    shared_loss = InfoNCE(negative_mode="mixed")
    private_loss = InfoNCE(negative_mode="paired")

    pos_idx = np.random.randint(num_cameras)
    anchor_idxs = [i for i in range(num_cameras) if i != pos_idx]
    multi_view_loss = 0

    for anchor_idx in anchor_idxs:
        shared_query = z_shared[:, anchor_idx]
        private_query = z_private[:, anchor_idx]

        shared_positive_key = z_target_shared[:, pos_idx]
        private_positive_key = next_z_target_private[:, anchor_idx]

        shared_negative_key = z_target_private
        private_negative_key = z_target_private[:, [pos_idx]]

        multi_view_loss += shared_loss(shared_query, shared_positive_key, shared_negative_key)
        multi_view_loss += private_loss(private_query, private_positive_key, private_negative_key)

    multi_view_loss /= len(anchor_idxs)

    agent.mvd_optimizer.zero_grad()
    multi_view_loss.backward()
    agent.mvd_optimizer.step()

    logger.log('train/mvd_loss', multi_view_loss, step)