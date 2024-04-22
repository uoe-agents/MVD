import numpy as np
import kornia
import torch
import torch.nn as nn
class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, image_pad, device, proprioceptive_state_shape=None):
        self.capacity = capacity
        self.device = device

        self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))

        if len(obs_shape)==1:
            self.obses = np.empty((capacity, *obs_shape), dtype=np.float32)
            self.next_obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        else:
            self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
            self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        self.episode = np.empty((capacity, 1), dtype=np.float32)

        if proprioceptive_state_shape is not None:
            self.collect_proprioceptive_state = True
            self.proprioceptive_states = np.empty((capacity, *proprioceptive_state_shape), dtype=np.float32)
        else:
            self.collect_proprioceptive_state = False

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max, episode, proprioceptive_state=None):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
        np.copyto(self.episode[self.idx], episode)

        if self.collect_proprioceptive_state:
            np.copyto(self.proprioceptive_states[self.idx], proprioceptive_state)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 (self.capacity - 1) if self.full else (self.idx - 1),
                                 size=batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        if self.collect_proprioceptive_state:
            proprioceptive_states = torch.as_tensor(self.proprioceptive_states[idxs], device=self.device)
            next_proprioceptive_states = torch.as_tensor(self.proprioceptive_states[idxs+1], device=self.device)
        else:
            proprioceptive_states = None
            next_proprioceptive_states = None

        return (obses, actions, rewards, next_obses, not_dones_no_max, proprioceptive_states, next_proprioceptive_states)

    def sample_drq(self, batch_size):
        idxs = np.random.randint(0,
                                 (self.capacity - 1) if self.full else (self.idx - 1),
                                 size=batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses_aug = obses.copy()
        next_obses_aug = next_obses.copy()

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        obses_aug = torch.as_tensor(obses_aug, device=self.device).float()
        next_obses_aug = torch.as_tensor(next_obses_aug, device=self.device).float()

        obses = self.aug_trans(obses)
        next_obses = self.aug_trans(next_obses)
        obses_aug = self.aug_trans(obses_aug)
        next_obses_aug = self.aug_trans(next_obses_aug)

        if self.collect_proprioceptive_state:
            proprioceptive_states = torch.as_tensor(self.proprioceptive_states[idxs], device=self.device)
            next_proprioceptive_states = torch.as_tensor(self.proprioceptive_states[idxs+1], device=self.device)
        else:
            proprioceptive_states = None
            next_proprioceptive_states = None

        return obses, actions, rewards, next_obses, not_dones_no_max, obses_aug, next_obses_aug, proprioceptive_states, next_proprioceptive_states