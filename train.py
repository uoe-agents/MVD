import collections
import os
import time
import torch
import numpy as np
import utils
from logger import Logger
from video import VideoRecorder
from algorithms.replay_buffer import ReplayBuffer
import algorithms
from arguments import parse_args

torch.backends.cudnn.benchmark = True

def make_env(cfg, cameras):
    if cfg.domain_name == "Panda":
        import gymnasium as gym
        import panda_gym
        env = gym.make(cfg.task_name)
        env = utils.MultiViewPanda(env, k=cfg.frame_stack, cameras=cameras, hw=cfg.image_size)

    elif cfg.domain_name == "MetaWorld":
        import metaworld
        mt1 = metaworld.MT1(cfg.task_name, seed=cfg.seed)
        env = mt1.train_classes[cfg.task_name](tasks=mt1.train_tasks, render_mode="rgb_array")
        env = utils.MultiViewMetaWorld(env, k=cfg.frame_stack, cameras=cameras)

    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.path.join(os.getcwd(), cfg.log_dir, cfg.exp_name, str(cfg.seed))
        assert not os.path.exists(self.work_dir), f'specified working directory {self.work_dir} already exists'
        os.makedirs(self.work_dir)
        print(f'workspace: {self.work_dir}')
        self.save_dir = os.path.join(self.work_dir, "trained_models")
        utils.write_info(cfg, os.path.join(self.work_dir, 'config.log'))
        self.cfg = cfg
        self.cameras = cfg.cameras

        self.logger = Logger(self.work_dir,
                             log_frequency=self.cfg.log_freq,
                             action_repeat=self.cfg.action_repeat,
                             eval_on_each_scenario=(self.cfg.eval_on_each_camera or len(self.cfg.cameras)==1),
                             domain_name=self.cfg.domain_name)
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        self.env = make_env(cfg, cameras=cfg.cameras)
        self.eval_env = make_env(cfg, cameras=cfg.cameras)

        self.env.reset(seed=cfg.seed)
        self.eval_env.reset(seed=cfg.seed+100)

        # seed for env reset
        self.rng = np.random.default_rng(seed=cfg.seed)

        action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]

        self.agent = algorithms.make_agent(self.env.observation_space.shape, self.env.action_space.shape, action_range, cfg, proprioceptive_state_shape=self.env.proprioceptive_state_shape if cfg.use_proprioceptive_state else None)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          cfg.replay_buffer_capacity,
                                          self.cfg.image_pad, self.device,
                                          proprioceptive_state_shape=self.env.proprioceptive_state_shape if cfg.use_proprioceptive_state else None)

        self.video_dir = os.path.join(self.work_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)

        self.video_recorder = VideoRecorder()

        self.step = 0

    def evaluate(self, record_video=False):
        average_episode_reward = 0
        successes = 0

        if record_video:
            self.video_recorder.init(enabled=True)

        for episode in range(self.cfg.num_eval_episodes):
            obs, info = self.eval_env.reset()

            done = False
            episode_reward = 0
            episode_step = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, proprioceptive_state=info.get("proprioceptive_state"), sample=False)

                obs, reward, terminated, truncated, info = self.eval_env.step(action)

                done = terminated or truncated
                if record_video:
                    self.video_recorder.record(self.eval_env)
                episode_reward += reward
                episode_step += 1

            average_episode_reward += episode_reward
            try:
                successes += info['success']
            except:
                successes += info['is_success']
            if record_video:
                self.video_recorder.save(os.path.join(self.video_dir, f'eval_{episode}.mp4'))
                self.video_recorder.reset()

        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward, self.step)

        success_rate = successes / self.cfg.num_eval_episodes
        self.logger.log('eval/success_rate', success_rate, self.step)
        self.logger.dump(self.step, save=True, ty="eval")

        if len(self.cameras) == 1:
            self.logger.log(f'eval_scenarios/{self.cameras[0]}_cam_episode_reward', average_episode_reward, self.step)
            self.logger.log(f'eval_scenarios/{self.cameras[0]}_cam_success_rate', success_rate, self.step)
            self.logger.dump(self.step, save=True, ty="eval_scenarios")

    def evaluate_on_each_scenario(self, record_video=False):
        for idx, cam in enumerate(self.cameras):
            average_episode_reward = 0
            successes = 0

            if record_video:
                self.video_recorder.init(enabled=True)

            for episode in range(self.cfg.num_eval_episodes):
                obs, info = self.eval_env.reset()
                obs = obs.reshape(len(self.cameras), -1, *obs.shape[1:])[idx]

                done = False
                episode_reward = 0
                episode_step = 0

                while not done:
                    with utils.eval_mode(self.agent):
                        action = self.agent.act(obs, proprioceptive_state=info.get("proprioceptive_state"), sample=False, eval_on_single_cam=True)

                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    obs = obs.reshape(len(self.cameras), -1, *obs.shape[1:])[idx]

                    done = terminated or truncated
                    if record_video:
                        self.video_recorder.record(self.eval_env, camera=cam)
                    episode_reward += reward
                    episode_step += 1

                average_episode_reward += episode_reward
                try:
                    successes += info['success']
                except:
                    successes += info['is_success']
                if record_video:
                    self.video_recorder.save(os.path.join(self.video_dir, f'eval_scenarios_{cam}_cam_{episode}.mp4'))
                    self.video_recorder.reset()

            average_episode_reward /= self.cfg.num_eval_episodes
            success_rate = successes / self.cfg.num_eval_episodes

            self.logger.log(f'eval_scenarios/{cam}_cam_episode_reward', average_episode_reward, self.step)
            self.logger.log(f'eval_scenarios/{cam}_cam_success_rate', success_rate, self.step)

        self.logger.dump(self.step, save=True, ty="eval_scenarios")

    def run(self):
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        start_time = time.time()

        total_num_steps = self.cfg.num_train_steps

        successes = collections.deque([], maxlen=10)

        while self.step <= (total_num_steps + 1):
            if done:
                if self.step>0:
                    try:
                        successes.append(info['success'])
                    except:
                        successes.append(info['is_success'])
                    self.logger.log('train/episode_reward', episode_reward, self.step)
                    self.logger.log('train/success_rate', np.mean(successes), self.step)
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps), ty="train")

                obs, info = self.env.reset()

                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # evaluate agent periodically
            if self.step % self.cfg.eval_freq == 0:
                self.logger.log('eval/episode', episode, self.step)
                self.evaluate(record_video=self.cfg.save_video)
                if self.cfg.eval_on_each_camera:
                    self.logger.log('eval_scenarios/episode', episode, self.step)
                    self.evaluate_on_each_scenario(record_video=self.cfg.save_video)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, proprioceptive_state=info.get("proprioceptive_state"), sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger, self.step)

            if self.step > 0 and self.step % self.cfg.save_freq == 0:
                saveables = {
                    "actor": self.agent.actor.state_dict(),
                    "critic": self.agent.critic.state_dict(),
                    "critic_target": self.agent.critic_target.state_dict()
                }
                save_at = os.path.join(self.save_dir, f"env_step{self.step * self.cfg.action_repeat}")
                os.makedirs(save_at, exist_ok=True)
                torch.save(saveables, os.path.join(save_at, "models.pt"))

            next_obs, reward, terminated, truncated, info = self.env.step(action)

            # allow infinite bootstrap
            done = terminated or truncated
            done_no_max = 0 if truncated else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max, episode, proprioceptive_state=info.get("proprioceptive_state"))

            obs = next_obs
            episode_step += 1
            self.step += 1

def main(cfg):
    from train import Workspace as W
    global workspace
    workspace = W(cfg)
    start_time = time.time()
    workspace.run()
    print("total run time: ", time.time()-start_time)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
