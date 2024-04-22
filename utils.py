import math
import os
import random
from collections import deque
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from datetime import datetime
import subprocess
import json
import glob
import pybullet as p
from torch.distributions.utils import _standard_normal
import re
import cv2


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # gain = nn.init.calculate_gain('relu')
        # nn.init.orthogonal_(m.weight.data, gain)
        # if hasattr(m.bias, 'data'):
        #     m.bias.data.fill_(0.0)
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()

class MultiViewMetaWorld(gym.Wrapper):
    def __init__(self, env, k, cameras):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._cameras = cameras
        self._frames = []
        for _ in cameras:
            self._frames.append(deque([], maxlen=k))
        self.hw = 84
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3 * k * len(cameras), self.hw, self.hw),
            dtype=env.observation_space.dtype)
        self._max_episode_steps = self.env.max_path_length
        self.proprioceptive_state_shape = (4,)

    def reset(self, seed=None):
        task = random.choice(self.env.tasks)
        self.env.set_task(task)
        obs, info = self.env.reset(seed=seed)

        # first 4 elements of obs contain the 3D end effector position and 1D measurement of how open the gripper is
        proprioceptive_info = obs[:4]

        for idx,cam in enumerate(self._cameras):
            if cam == "first_person":
                img = self.env.mujoco_renderer.render("rgb_array", camera_name="first_person")
            elif cam == "third_person":
                img = self.env.mujoco_renderer.render("rgb_array", camera_name="third_person")[::-1,:,:]
            else:
                raise Exception(f"camera {cam} not supported")
            img = cv2.resize(img, (self.hw, self.hw), interpolation=cv2.INTER_AREA)
            img = img.transpose(2, 0, 1).copy()

            for k in range(self._k):
                self._frames[idx].append(img)

        info["proprioceptive_state"] = proprioceptive_info

        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # first 4 elements of obs contain the 3D end effector position and 1D measurement of how open the gripper is
        proprioceptive_info = obs[:4]

        for idx,cam in enumerate(self._cameras):
            if cam == "first_person":
                img = self.env.mujoco_renderer.render("rgb_array", camera_name="first_person")
            elif cam == "third_person":
                img = self.env.mujoco_renderer.render("rgb_array", camera_name="third_person")[::-1,:,:]
            else:
                raise Exception(f"camera {cam} not supported")
            img = cv2.resize(img, (self.hw, self.hw), interpolation=cv2.INTER_AREA)
            img = img.transpose(2, 0, 1).copy()

            self._frames[idx].append(img)

        info["proprioceptive_state"] = proprioceptive_info

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        assert len(self._frames) == len(self._cameras)
        output = []
        for frames in self._frames:
            assert len(frames) == self._k
            output.append(np.concatenate(frames, axis=0))
        return np.concatenate(output, axis=0)

    def render(self, mode="rgb_array", height=480, width=480, camera=None):
        if camera is None:
            camera = "third_person"

        if camera == "first_person":
            img = self.env.mujoco_renderer.render("rgb_array", camera_name="first_person")
        elif camera == "third_person":
            img = self.env.mujoco_renderer.render("rgb_array", camera_name="third_person")[::-1, :, :]
        else:
            raise Exception(f"camera {camera} not supported")

        if height!=480 or width!=480:
            img = cv2.resize(img, (height, width), interpolation=cv2.INTER_AREA)

        return img
class MultiViewPanda(gym.Wrapper):
    def __init__(self, env, k, cameras, hw):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._cameras = cameras
        self._frames = []
        for _ in cameras:
            self._frames.append(deque([], maxlen=k))
        self.hw = hw
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3 * k * len(cameras), hw, hw),
            dtype=np.uint8)
        self._max_episode_steps = env._max_episode_steps
        self.proprioceptive_state_shape = (6,)

        self.projection_matrix = self.env.unwrapped.sim.physics_client.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
            nearVal=0.01,
            farVal=1000.
        )

        self.cam_img_args = dict(
            width=self.hw,
            height=self.hw,
            projectionMatrix=self.projection_matrix,
            shadow=1,
            renderer=self.env.unwrapped.sim.physics_client.ER_BULLET_HARDWARE_OPENGL,
            flags=self.env.unwrapped.sim.physics_client.ER_NO_SEGMENTATION_MASK,
        )

        self.p = self.env.unwrapped.sim.physics_client
        self.cid = self.env.unwrapped.sim.physics_client._client

        j = 0
        self._dofs = []
        self._joint_name_to_i_dof = {}
        self._joint_name_to_i_joint = {}
        self._ll = []
        self._ul = []
        self._jd = []
        for i in range(self.p.getNumJoints(0)):
            joint_info = self.p.getJointInfo(0, i, physicsClientId=self.cid)

            joint_name = joint_info[1].decode()
            joint_type = joint_info[2]
            self._joint_name_to_i_joint[joint_name] = i
            if joint_type in [p.JOINT_PRISMATIC, p.JOINT_REVOLUTE]:
                self._dofs.append(i)
                self._jd.append(joint_info[6])
                self._ll.append(joint_info[8])
                self._ul.append(joint_info[9])

                self._joint_name_to_i_dof[joint_name] = j
                j += 1

    def get_ego_view_matrix(self, eye_offset=0, target_offset=0, up_offset=0):
        ee_state = self.env.unwrapped.sim.physics_client.getLinkState(0, 11)
        ee_pos, ee_ori = np.array(ee_state[0]), np.array(ee_state[1])
        ee_R = self.env.unwrapped.sim.physics_client.getMatrixFromQuaternion(ee_ori)
        ee_R = np.array(ee_R).reshape(3, 3)
        ee_x, ee_y, ee_z = ee_R[:, 0], ee_R[:, 1], ee_R[:, 2]
        eye = ee_pos - 0.08 * ee_z - 0.15 * ee_x
        target = ee_pos + 0.1 * ee_z
        view_matrix = self.env.unwrapped.sim.physics_client.computeViewMatrix(
            cameraEyePosition=eye + eye_offset,
            cameraTargetPosition=target + target_offset,
            cameraUpVector=ee_x + up_offset
        )
        return view_matrix

    def get_proprioceptive_state(self):
        ee_pos = self.env.unwrapped.robot.get_ee_position()
        ee_grip = np.array([self.env.unwrapped.robot.get_fingers_width()])
        base_pos, _ = p.getBasePositionAndOrientation(0, self.cid)
        ee_pos_rel_base = ee_pos - base_pos
        fj1_contacts = p.getContactPoints(0, linkIndexA=self._joint_name_to_i_joint['panda_finger_joint1'],
                                          physicsClientId=self.cid)
        fj2_contacts = p.getContactPoints(0, linkIndexA=self._joint_name_to_i_joint['panda_finger_joint2'],
                                          physicsClientId=self.cid)
        contact_flags = np.array([1. if len(c) > 0 else -1. for c in [fj1_contacts, fj2_contacts]])

        proprioceptive_info = np.concatenate((ee_pos_rel_base, ee_grip, contact_flags), axis=-1)
        return proprioceptive_info

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)

        for idx,cam in enumerate(self._cameras):
            if cam == "first_person":
                view_matrix = self.get_ego_view_matrix()
                self.cam_img_args['viewMatrix'] = view_matrix
                w, h, im_rgba, im_d, im_seg = self.env.unwrapped.sim.physics_client.getCameraImage(**self.cam_img_args)
                img = np.transpose(im_rgba[:, :, :3], (2, 0, 1))
            elif cam == "third_person_front":
                view_matrix = self.env.unwrapped.sim.physics_client.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=[-0.2, 0, 0],
                    distance=1.0,
                    yaw=90,
                    pitch=-20,
                    roll=0,
                    upAxisIndex=2
                )
                self.cam_img_args['viewMatrix'] = view_matrix
                w, h, im_rgba, im_d, im_seg = self.env.unwrapped.sim.physics_client.getCameraImage(**self.cam_img_args)
                img = np.transpose(im_rgba[:, :, :3], (2, 0, 1))
            elif cam == "third_person_side":
                view_matrix = self.env.unwrapped.sim.physics_client.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=[0.1, -0.3, 0.23],
                    distance=0.41,
                    yaw=35,
                    pitch=-32,
                    roll=0,
                    upAxisIndex=2
                )
                self.cam_img_args['viewMatrix'] = view_matrix
                w, h, im_rgba, im_d, im_seg = self.env.unwrapped.sim.physics_client.getCameraImage(**self.cam_img_args)
                img = np.transpose(im_rgba[:, :, :3], (2, 0, 1))
            else:
                raise AssertionError(f"camera {cam} not implemented")

            for k in range(self._k):
                self._frames[idx].append(img)

        info["proprioceptive_state"] = self.get_proprioceptive_state()

        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        for idx,cam in enumerate(self._cameras):
            if cam == "first_person":
                view_matrix = self.get_ego_view_matrix()
                self.cam_img_args['viewMatrix'] = view_matrix
                w, h, im_rgba, im_d, im_seg = self.env.unwrapped.sim.physics_client.getCameraImage(**self.cam_img_args)
                img = np.transpose(im_rgba[:, :, :3], (2, 0, 1))
            elif cam == "third_person_front":
                view_matrix = self.env.unwrapped.sim.physics_client.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=[-0.2, 0, 0],
                    distance=1.0,
                    yaw=90,
                    pitch=-20,
                    roll=0,
                    upAxisIndex=2
                )
                self.cam_img_args['viewMatrix'] = view_matrix
                w, h, im_rgba, im_d, im_seg = self.env.unwrapped.sim.physics_client.getCameraImage(**self.cam_img_args)
                img = np.transpose(im_rgba[:, :, :3], (2, 0, 1))
            elif cam == "third_person_side":
                view_matrix = self.env.unwrapped.sim.physics_client.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=[0.1, -0.3, 0.23],
                    distance=0.41,
                    yaw=35,
                    pitch=-32,
                    roll=0,
                    upAxisIndex=2
                )
                self.cam_img_args['viewMatrix'] = view_matrix
                w, h, im_rgba, im_d, im_seg = self.env.unwrapped.sim.physics_client.getCameraImage(**self.cam_img_args)
                img = np.transpose(im_rgba[:, :, :3], (2, 0, 1))
            else:
                raise AssertionError(f"camera {cam} not implemented")

            self._frames[idx].append(img)

        info["proprioceptive_state"] = self.get_proprioceptive_state()

        # do not end episode after success
        return self._get_obs(), reward, False, truncated, info

    def _get_obs(self):
        assert len(self._frames) == len(self._cameras)
        output = []
        for frames in self._frames:
            assert len(frames) == self._k
            output.append(np.concatenate(frames, axis=0))
        return np.concatenate(output, axis=0)

    def render(self, mode="rgb_array", height=480, width=480, camera=None):
        render_img_args = dict(
            width=width,
            height=height,
            projectionMatrix=self.projection_matrix,
            shadow=1,
            renderer=self.env.unwrapped.sim.physics_client.ER_BULLET_HARDWARE_OPENGL,
            flags=self.env.unwrapped.sim.physics_client.ER_NO_SEGMENTATION_MASK,
        )

        if camera == "first_person":
            view_matrix = self.get_ego_view_matrix()
            render_img_args['viewMatrix'] = view_matrix
            w, h, im_rgba, im_d, im_seg = self.env.unwrapped.sim.physics_client.getCameraImage(**render_img_args)
        elif camera == "third_person_front":
            view_matrix = self.env.unwrapped.sim.physics_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[-0.2, 0, 0],
                distance=1.0,
                yaw=90,
                pitch=-20,
                roll=0,
                upAxisIndex=2
            )
            render_img_args['viewMatrix'] = view_matrix
            w, h, im_rgba, im_d, im_seg = self.env.unwrapped.sim.physics_client.getCameraImage(**render_img_args)
            # im_rgba = self.env.unwrapped.sim.render(width=width, height=height, target_position=[-0.2, 0, 0],
            #                                         distance=1.0, yaw=90, pitch=-20, roll=0)
        elif camera == "third_person_side":
            view_matrix = self.env.unwrapped.sim.physics_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0.1, -0.3, 0.23],
                distance=0.41,
                yaw=35,
                pitch=-32,
                roll=0,
                upAxisIndex=2
            )
            render_img_args['viewMatrix'] = view_matrix
            w, h, im_rgba, im_d, im_seg = self.env.unwrapped.sim.physics_client.getCameraImage(**render_img_args)
        else:
            im_rgba = self.env.render()

        img = im_rgba[:, :, :3]

        return img

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))

class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

def write_info(args, fp):
    try:
        data = {
            'timestamp': str(datetime.now()),
            'git': subprocess.check_output(["git", "describe", "--always"]).strip().decode(),
            'args': vars(args)
            }
    except:
        data = {
            'timestamp': str(datetime.now()),
            'args': vars(args)
        }
    with open(fp, 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ': '))

def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs

def postprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    # bins = 2**bits
    # assert obs.dtype == torch.float32
    # obs = obs + 0.5
    # obs = (obs * bins) - torch.rand_like(obs)
    # obs = obs * bins
    # if bits < 8:
    #     obs * (2**(8-bits))
    obs = (obs + 0.5) * 255
    return obs


def listdir(dir_path, filetype='jpg', sort=True):
    fpath = os.path.join(dir_path, f'*.{filetype}')
    fpaths = glob.glob(fpath, recursive=True)
    if sort:
        return sorted(fpaths)
    return

class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)

def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)