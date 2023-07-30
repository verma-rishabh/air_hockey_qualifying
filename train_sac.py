# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import tqdm
import sys
import time
from distutils.util import strtobool
import copy
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from stable_baselines3.common.buffers import ReplayBuffer
from utils import ReplayBuffer
from baseline.baseline_agent.baseline_agent import build_agent
from omegaconf import OmegaConf

import os
import sys
from torch.utils.tensorboard.writer import SummaryWriter
from air_hockey_challenge.framework.agent_base import AgentBase
from air_hockey_challenge.framework import AirHockeyChallengeWrapper
sys.path.append('./air_hockey_agent')
from air_hockey_agent.sac_agent import SAC_Agent, SoftQNetwork, Actor
from air_hockey_challenge.framework.evaluate_agent import evaluate

from air_hockey_agent.agent_builder import build_agent as sac_build

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--env-id", type=str, default="7dof-hit",
        help="the id of the environment")
    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=int(1e6),
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.01,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=1024,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=5000,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=1e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=3e-4, #1e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=1,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=2, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--alpha", type=float, default=0.0008,
            help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id,seed):
    env = AirHockeyChallengeWrapper(env_id)
    env.env_info["rl_info"].observation_space.seed = seed
    return env

def reward_mushroomrl_constr(self, next_state, action):
        """
        taking constraint violations innto account
        """
        # Get the joint position and velocity from the observation
        q = next_state[self.env_info['joint_pos_ids']]
        dq = next_state[self.env_info['joint_vel_ids']]

        constraints = ["joint_pos_constr", "joint_vel_constr", "ee_constr", "link_constr"]
        pos_constr = self.env_info['constraints'].get("joint_pos_constr").fun(q, dq)
        vel_constr = self.env_info['constraints'].get("joint_vel_constr").fun(q, dq)
        ee_constr = self.env_info['constraints'].get("ee_constr").fun(q, dq)
        link_constr = self.env_info['constraints'].get("link_constr").fun(q, dq)
        
        pos_err = np.sum(pos_constr[pos_constr > 0]) if np.any(pos_constr > 0) else 0
        vel_err = np.sum(vel_constr[vel_constr > 0]) if np.any(vel_constr > 0) else 0
        ee_err = np.sum(ee_constr[ee_constr > 0]) if np.any(ee_constr > 0) else 0
        link_err = np.sum(link_constr[link_constr > 0]) if np.any(link_constr > 0) else 0

        constraint_reward = - (pos_err + vel_err + ee_err + link_err)

        # if constraint_reward !=0 :
            # print("constraint reward", constraint_reward)

        r = 0
        mod_next_state = next_state # changing frame of puck pos (wrt origin)
        mod_next_state[:3]  = mod_next_state[:3] - [1.51,0,0.1]
        absorbing = self.base_env.is_absorbing(mod_next_state)
        puck_pos, puck_vel = self.base_env.get_puck(mod_next_state)                     # extracts from obs therefore robot frame

        goal = np.array([0.974, 0])
        effective_width = 0.519 - 0.03165

        # Calculate bounce point by assuming incoming angle = outgoing angle
        w = (abs(puck_pos[1]) * goal[0] + goal[1] * puck_pos[0] - effective_width * puck_pos[
            0] - effective_width *
                goal[0]) / (abs(puck_pos[1]) + goal[1] - 2 * effective_width)


        side_point = np.array([w, np.copysign(effective_width, puck_pos[1])])
        #print("side_point",side_point)

        vec_puck_side = (side_point - puck_pos[:2]) / np.linalg.norm(side_point - puck_pos[:2])
        vec_puck_goal = (goal - puck_pos[:2]) / np.linalg.norm(goal - puck_pos[:2])
        has_hit = self.base_env._check_collision("puck", "robot_1/ee")    
        
        # If puck is out of bounds
        if absorbing:
            # If puck is in the opponent goal
            if (puck_pos[0] - self.env_info['table']['length'] / 2) > 0 and \
                    (np.abs(puck_pos[1]) - self.env_info['table']['goal_width']) < 0:
                    # print("puck_pos",puck_pos,"absorbing",absorbing)
                r = 200

        else:
            if not has_hit:
                ee_pos = self.base_env.get_ee()[0]                                     # tO check
                # print(ee_pos,self.policy.get_ee_pose(next_state)[0] - [1.51,0,0.1])

                dist_ee_puck = np.linalg.norm(puck_pos[:2] - ee_pos[:2])                # changing to 2D plane because used to normalise 2D vector

                vec_ee_puck = (puck_pos[:2] - ee_pos[:2]) / dist_ee_puck

                cos_ang_side = np.clip(vec_puck_side @ vec_ee_puck, 0, 1)

                # Reward if vec_ee_puck and vec_puck_goal have the same direction
                cos_ang_goal = np.clip(vec_puck_goal @ vec_ee_puck, 0, 1)
                cos_ang = np.max([cos_ang_goal, cos_ang_side])

                r = np.exp(-8 * (dist_ee_puck - 0.08)) * cos_ang ** 2
            else:
                r_hit = 0.25 + min([1, (0.25 * puck_vel[0] ** 4)])

                r_goal = 0
                if puck_pos[0] > 0.7:
                    sig = 0.1
                    r_goal = 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((puck_pos[1] - 0) / sig, 2.) / 2)

                r = 2 * r_hit + 10 * r_goal

        r -= 1e-3 * np.linalg.norm(action)
        r += (constraint_reward/10)

        return r

def reward_mushroomrl(env,next_state, action):
    
    r = 0
    mod_next_state = next_state                            # changing frame of puck pos (wrt origin)
    mod_next_state[:3]  = mod_next_state[:3] - [1.51,0,0.1]
    absorbing = env.base_env.is_absorbing(copy.deepcopy(mod_next_state))
    puck_pos, puck_vel = env.base_env.get_puck(copy.deepcopy(mod_next_state))

    q = next_state[env.env_info['joint_pos_ids']]
    dq = next_state[env.env_info['joint_vel_ids']]
    constraints = env.env_info['constraints'].keys()

    constraint_reward = 0
    for constr in constraints:
        error = env.env_info['constraints'].get(constr).fun(q, dq)
        constr_error = np.sum(error[error > 0]) if np.any(error > 0) else 0
        constraint_reward -= constr_error
        
    # constraint_reward *= 300 

    if constraint_reward !=0:
        print("constraint reward", constraint_reward)

    ###################################################
    goal = np.array([0.974, 0])
    effective_width = 0.519 - 0.03165

    # Calculate bounce point by assuming incoming angle = outgoing angle
    w = (abs(puck_pos[1]) * goal[0] + goal[1] * puck_pos[0] - effective_width * puck_pos[
        0] - effective_width *
            goal[0]) / (abs(puck_pos[1]) + goal[1] - 2 * effective_width)


    side_point = np.array([w, np.copysign(effective_width, puck_pos[1])])
    #print("side_point",side_point)

    vec_puck_side = (side_point - puck_pos[:2]) / np.linalg.norm(side_point - puck_pos[:2])
    vec_puck_goal = (goal - puck_pos[:2]) / np.linalg.norm(goal - puck_pos[:2])
    has_hit = env.base_env._check_collision("puck", "robot_1/ee")

    # If puck is out of bounds
    if absorbing:
        # If puck is in the opponent goal
        if (puck_pos[0] - env.env_info['table']['length'] / 2) > 0 and \
                (np.abs(puck_pos[1]) - env.env_info['table']['goal_width']) < 0:
                # print("puck_pos",puck_pos,"absorbing",absorbing)
                r = 200
    else:
        if not has_hit:
            ee_pos = env.base_env.get_ee()[0]                                     # tO check
            # print(ee_pos,self.policy.get_ee_pose(next_state)[0] - [1.51,0,0.1])

            dist_ee_puck = np.linalg.norm(puck_pos[:2] - ee_pos[:2])                # changing to 2D plane because used to normalise 2D vector

            vec_ee_puck = (puck_pos[:2] - ee_pos[:2]) / dist_ee_puck

            cos_ang_side = np.clip(vec_puck_side @ vec_ee_puck, 0, 1)

            # Reward if vec_ee_puck and vec_puck_goal have the same direction
            cos_ang_goal = np.clip(vec_puck_goal @ vec_ee_puck, 0, 1)
            cos_ang = np.max([cos_ang_goal, cos_ang_side])

            r = np.exp(-8 * (dist_ee_puck - 0.08)) * cos_ang ** 2
        else:
            r_hit = 0.25 + min([1, (0.25 * puck_vel[0] ** 4)])

            r_goal = 0
            if puck_pos[0] > 0.7:
                sig = 0.1
                r_goal = 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((puck_pos[1] - 0) / sig, 2.) / 2)

            r = 2 * r_hit + 10 * r_goal

    r -= 1e-3 * np.linalg.norm(action)
    
    des_z = env.base_env.env_info['robot']['ee_desired_height']
    tolerance = 0.02

    # if abs(actor.get_ee_pose(copy.deepcopy(next_state))[0][1])>0.519:  # should replace with env variables some day
    #     r -=0.1 
    # if (actor.get_ee_pose(copy.deepcopy(next_state))[0][0])<0.536:
    #     r -=0.1 
    # if (actor.get_ee_pose(copy.deepcopy(next_state))[0][2])<des_z-tolerance*10 or (actor.get_ee_pose(copy.deepcopy(next_state))[0][2])>des_z+tolerance*10:
    #     r -=0.1
    r += constraint_reward
    return r


if __name__ == "__main__":
    args = parse_args()
    # args = OmegaConf.load('train_sac.yaml')
    # print(args.env_id)
    timestamp = time.time()
    formatted_time = time.strftime("%d-%m-%Y %H:%M", time.localtime(timestamp))

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{formatted_time}"

    writer = SummaryWriter(f"runs/sac/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # env = make_env(args.env_id, args.seed)
    env = AirHockeyChallengeWrapper(args.env_id)
    env.env_info["rl_info"].observation_space.seed = args.seed

    env_info = env.env_info

    state_dim = env_info["rl_info"].observation_space.low.shape[0]
    state_dim_ = env_info["rl_info"].observation_space.low.shape

    action_dim = env_info["rl_info"].action_space.low.shape[0]
    action_dim_ = (action_dim,)
    # state, done = env.reset(), False #initial_state

    action_space = np.concatenate((env_info["robot"]["joint_pos_limit"], 
                                  env_info["robot"]["joint_vel_limit"]), axis=1)

    # max_action = float(env.single_action_space.high[0])
    max_action = action_space[1,:]
    min_action = action_space[0,:]

    agent = SAC_Agent(env_info)
    base_agent = build_agent(env_info)

    actor = agent.actor.to(device)
    qf1 = agent.qf1.to(device)
    qf2 = agent.qf2.to(device)
    qf1_target = agent.qf1_target.to(device)
    qf2_target = agent.qf2_target.to(device)

    # qf1_target.load_state_dict(qf1.state_dict())
    # qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(env.env_info["rl_info"].action_space.low.shape)).to(device).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # env.single_observation_space.dtype = np.float32
    # env_info["rl_info"].action_space.low.dtype = np.float32
    # env_info["rl_info"].action_space.high.dtype = np.float32

    # the offline buffer ------------------------------------------
    obs, done = env.reset(),False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    intermediate_t=0
    rb_offline = ReplayBuffer(state_dim, action_dim, max_size=100000)
    print("populating the offline buffer:.......")
    for i in range(10000):
        action = base_agent.draw_action(np.array(obs))
        next_obs, reward_, done, info = env.step(action)
        # env.render()
        next_obs_copy = copy.deepcopy(next_obs)
        reward = reward_mushroomrl_constr(env,next_obs, action)
        success = info["success"]
        episode_reward += reward
        obs = next_obs_copy
        rb_offline.add(obs, action.flatten(),next_obs_copy,reward,done)

        if done or intermediate_t > 300:
             
            obs, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 
            intermediate_t=0

    # the offline buffer ------------------------------------------

    # the online buffer ------------------------------------------

    rb_online = ReplayBuffer(state_dim, action_dim, max_size=args.buffer_size)
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, done = env.reset(),False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    intermediate_t=0

    for global_step in range(args.total_timesteps):

        episode_timesteps += 1
        intermediate_t +=1

        if global_step < args.learning_starts: # base agent instead of random actions
            
            # if global_step%2==0:
                # action = base_agent.draw_action(np.array(obs))
            # else:
            action = torch.Tensor([random.uniform(min_action[i] * 0.95 , max_action[i]* 0.95) for i in range(action_dim)]).reshape(2,7)
        else:
            action, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            
        # action = torch.Tensor(action) #.to(device).detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward_, done, info = env.step(action)
        # env.render()
        next_obs_copy = copy.deepcopy(next_obs)
        reward = reward_mushroomrl_constr(env,next_obs, action)
        writer.add_scalar("charts/rewards",reward, global_step)
        # print(reward)
        success = info["success"]
        episode_reward += reward
        obs = next_obs_copy
        # not_dones = 1 - dones 
    
        rb_online.add(obs, action.flatten(),next_obs_copy,reward,done)

        if done or intermediate_t > 300:

            print(f"global_step={global_step},Episode Num: {episode_num+1}, episodic_return={episode_reward:.3f}")
            writer.add_scalar("charts/episodic_reward",episode_reward, global_step)
            writer.add_scalar("charts/episodic_length", episode_timesteps, global_step)
             
            obs, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 
            intermediate_t=0

        if global_step > args.learning_starts:
            data1 = rb_online.sample(int(args.batch_size/2))
            data2 = rb_offline.sample(int(args.batch_size/2))
            data = (
                    torch.cat((data1[0], data2[0]), dim=0),
                    torch.cat((data1[1], data2[1]), dim=0),
                    torch.cat((data1[2], data2[2]), dim=0),
                    torch.cat((data1[3], data2[3]), dim=0),
                    torch.cat((data1[4], data2[4]), dim=0)
                    )

            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data[2])
                # next_state_actions, next_state_log_pi, _ = torch.Tensor(np.array([actor.get_action(data[2][i,:]) for i in range(data[2].shape[0])]))
                qf1_next_target = qf1_target(data[2].to(device), torch.Tensor(next_state_actions).to(device))
                qf2_next_target = qf2_target(data[2].to(device), torch.Tensor(next_state_actions).to(device))
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                # print(data[4].shape)
                next_q_value = data[3]+ data[4] * args.gamma * (min_qf_next_target)

            qf1_a_values = qf1(data[0], data[1])
            qf2_a_values = qf2(data[0], data[1])
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data[0])
                    qf1_pi = qf1(torch.Tensor(data[0]).to(device), torch.Tensor(pi).to(device))
                    qf2_pi = qf2(torch.Tensor(data[0]).to(device),torch.Tensor(pi).to(device))
                    min_qf_pi = torch.min(qf1_pi, qf2_pi) #.view(-1)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
            
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data[0])
                        alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                
                print(f"global step....{global_step}")
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
            
            if global_step % 6000 == 0:
                agent.save(f"./models/sac_agent/sac")
                print(f"model saved, evaluating:.....global_step = {global_step}")
                evaluate(sac_build, "./logs", ["7dof-hit"], n_episodes=5, generate_score=None,
                        quiet=True, render=False, interpolation_order=3)
                
    
    agent.save(f"./models/sac_agent/sac")
    print("model saved at")
    env.stop()
    writer.close()