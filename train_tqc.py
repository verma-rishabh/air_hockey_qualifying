import numpy as np
import torch
import argparse
import os
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from air_hockey_agent.agent_builder_tqc import build_agent
from utils import ReplayBuffer, solve_hit_config_ik_null
from torch.utils.tensorboard.writer import SummaryWriter
from omegaconf import OmegaConf
# from air_hockey_challenge.utils.kinematics import inverse_kinematics, jacobian
from datetime import datetime
import copy
from reward import HitReward, DefendReward, PrepareReward

class train(AirHockeyChallengeWrapper):
    def __init__(self, env=None, custom_reward_function=HitReward(), interpolation_order=3, **kwargs):
        # Load config file
        self.conf = OmegaConf.load('train_tqc.yaml')
        env = self.conf.env
        # base env
        super().__init__(env, custom_reward_function,interpolation_order, **kwargs)
        # seed
        self.seed(self.conf.agent.seed)
        torch.manual_seed(self.conf.agent.seed)
        np.random.seed(self.conf.agent.seed)
        # env variables
        self.action_shape = self.env_info["rl_info"].action_space.shape[0]
        # self.action_shape = 3
        self.observation_shape = self.env_info["rl_info"].observation_space.shape[0]
        # policy
        self.policy = build_agent(self.env_info)
        # action_space.high
        pos_max = self.env_info['robot']['joint_pos_limit'][1]
        vel_max = self.env_info['robot']['joint_vel_limit'][1] 
        max_ = np.stack([pos_max,vel_max])
        self.max_action  = max_.reshape(14,)
        # self.min_action = np.array([0.65,-0.40,0])
        # self.max_action = np.array([1.32,0.40,1.5])
        # make dirs 
        self.make_dir()
        self.tensorboard = SummaryWriter(self.conf.agent.dump_dir + "/tensorboard/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        # load model if defined
        if self.conf.agent.load_model!= "":
            policy_file = self.conf.agent.file_name if self.conf.agent.load_model == "default" else self.conf.agent .load_model
            print("loading model from file: ", policy_file)
            self.policy.load(self.conf.agent.dump_dir + f"/models/{policy_file}")
        
        self.replay_buffer = ReplayBuffer(self.observation_shape, self.action_shape)
    
    def _step(self,state,action):
        action = self.policy.action_scaleup(action)                                     # [-1,1] -> [-self.max_action,self.max_action]
        next_state, reward, done, info = self.step(action)
        reward += self.reward_mushroomrl(copy.deepcopy(next_state),copy.deepcopy(action)) 

        return next_state, reward, done, info

    
    def make_dir(self):
        if not os.path.exists(self.conf.agent.dump_dir+"/results"):
            os.makedirs(self.conf.agent.dump_dir+"/results")

        if not os.path.exists(self.conf.agent.dump_dir+"/models"):
            os.makedirs(self.conf.agent.dump_dir+"/models")
    
    def reward_mushroomrl(self, next_state, action):

        r = 0
    #     mod_next_state = next_state                            # changing frame of puck pos (wrt origin)
    #     mod_next_state[:3]  = mod_next_state[:3] - [1.51,0,0.1]
    #     absorbing = self.base_env.is_absorbing(mod_next_state)
    #     puck_pos, puck_vel = self.base_env.get_puck(mod_next_state)                     # extracts from obs therefore robot frame


    #     ###################################################
    #     goal = np.array([0.974, 0])
    #     effective_width = 0.519 - 0.03165

    #     # Calculate bounce point by assuming incoming angle = outgoing angle
    #     w = (abs(puck_pos[1]) * goal[0] + goal[1] * puck_pos[0] - effective_width * puck_pos[
    #         0] - effective_width *
    #          goal[0]) / (abs(puck_pos[1]) + goal[1] - 2 * effective_width)


    #     side_point = np.array([w, np.copysign(effective_width, puck_pos[1])])
    #     #print("side_point",side_point)

    #     vec_puck_side = (side_point - puck_pos[:2]) / np.linalg.norm(side_point - puck_pos[:2])
    #     vec_puck_goal = (goal - puck_pos[:2]) / np.linalg.norm(goal - puck_pos[:2])
    #     has_hit = self.base_env._check_collision("puck", "robot_1/ee")

        
    #     ###################################################
        
        

    #     # If puck is out of bounds
    #     if absorbing:
    #         # If puck is in the opponent goal
    #         if (puck_pos[0] - self.env_info['table']['length'] / 2) > 0 and \
    #                 (np.abs(puck_pos[1]) - self.env_info['table']['goal_width']) < 0:
    #                 print("puck_pos",puck_pos,"absorbing",absorbing)
    #                 r = 200

    #     else:
    #         if not has_hit:
    #             ee_pos = self.base_env.get_ee()[0]                                     # tO check
    #             # print(ee_pos,self.policy.get_ee_pose(next_state)[0] - [1.51,0,0.1])

    #             dist_ee_puck = np.linalg.norm(puck_pos[:2] - ee_pos[:2])                # changing to 2D plane because used to normalise 2D vector

    #             vec_ee_puck = (puck_pos[:2] - ee_pos[:2]) / dist_ee_puck

    #             cos_ang_side = np.clip(vec_puck_side @ vec_ee_puck, 0, 1)

    #             # Reward if vec_ee_puck and vec_puck_goal have the same direction
    #             cos_ang_goal = np.clip(vec_puck_goal @ vec_ee_puck, 0, 1)
    #             cos_ang = np.max([cos_ang_goal, cos_ang_side])

    #             r = np.exp(-8 * (dist_ee_puck - 0.08)) * cos_ang ** 2
    #         else:
    #             r_hit = 0.25 + min([1, (0.25 * puck_vel[0] ** 4)])

    #             r_goal = 0
    #             if puck_pos[0] > 0.7:
    #                 sig = 0.1
    #                 r_goal = 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((puck_pos[1] - 0) / sig, 2.) / 2)

    #             r = 2 * r_hit + 10 * r_goal

    #     r -= 1e-3 * np.linalg.norm(action)
        
        des_z = self.env_info['robot']['ee_desired_height']
        tolerance = 0.02

        if abs(self.policy.get_ee_pose(next_state)[0][1])>0.519:         # should replace with env variables some day
            r -=0.1 
        if (self.policy.get_ee_pose(next_state)[0][0])<0.536:
            r -=0.1 
        if (self.policy.get_ee_pose(next_state)[0][2])<des_z-tolerance or (self.policy.get_ee_pose(next_state)[0][2])>des_z+tolerance:
            r -=0.1
        return r



# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
    def eval_policy(self,t,eval_episodes=10):
        self.policy.actor.eval()
        for _ in range(eval_episodes):
            avg_reward = 0.
            # print(_)
            state, done = self.reset(), False
            episode_timesteps=0
            while not done and episode_timesteps<100:
                # print("ep",episode_timesteps)

                action = self.policy.select_action(state)
                next_state, reward, done, info = self._step(state,action)
                self.render()
                avg_reward += reward
                episode_timesteps+=1
                state = next_state
            self.tensorboard.add_scalar("eval_reward", avg_reward,t+_)
        self.policy.actor.train()
    # def _step(self,action):
    #     next_state, reward, done, info = self.step(action)
    #     reward = self.reward_mushroomrl(next_state, action) 
    #     return next_state, reward, done, info

    def train_model(self):
        state, done = self.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        # self.policy.actor.train()
        for t in range(int(self.conf.agent.max_timesteps)):
            self.policy.actor.train()
            critic_loss = np.nan
            actor_loss = np.nan
            alpha_loss = np.nan

            episode_timesteps += 1
           
            # Select action randomly or according to policy
            if t < self.conf.agent.start_timesteps:
                action = np.random.uniform(-self.max_action,self.max_action,(self.action_shape,)).reshape(2,7)
            else:
                action = self.policy.select_action(state)
            # Perform action
            next_state, reward, done, _ = self._step(state,action) 
            # self.render()
            done_bool = float(done) if episode_timesteps < self.conf.agent.max_episode_steps else 0   ###MAX EPISODE STEPS
            # Store data in replay buffer
            # print(action)
            self.replay_buffer.add(state, action.reshape(-1,), next_state, reward, done_bool)
            state = next_state
            episode_reward += reward

            # # Train agent after collecting sufficient data
            if t >= self.conf.agent.start_timesteps:
                actor_loss,critic_loss,alpha_loss=self.policy.train(self.replay_buffer, self.conf.agent.batch_size)

            if done or episode_timesteps > self.conf.agent.max_episode_steps: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward.sum():.3f}")
                # Reset environment
                if (actor_loss is not np.nan):
                    self.tensorboard.add_scalar("actor loss", actor_loss, t)
                if (critic_loss is not np.nan):
                    self.tensorboard.add_scalar("critic loss", critic_loss, t)
                if (alpha_loss is not np.nan):
                    self.tensorboard.add_scalar("alpha loss", alpha_loss, t)
                self.tensorboard.add_scalar("reward", episode_reward, t)
                state, done = self.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 
                
            if (t + 1) % self.conf.agent.eval_freq == 0:
                self.eval_policy(t)
                self.policy.save(self.conf.agent.dump_dir + f"/models/{self.conf.agent.file_name}")

x = train()
x.train_model()
