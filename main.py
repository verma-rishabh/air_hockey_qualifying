import numpy as np
import torch
import gym
import argparse
import os
import copy
from pathlib import Path

from torch.utils.tensorboard.writer import SummaryWriter
from tqc import structures, DEVICE
from tqc.trainer import Trainer
from tqc.structures import Actor, Critic, RescaleAction
from tqc.functions import eval_policy
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from gym import spaces
from datetime import datetime
import utils
from air_hockey_challenge.utils.kinematics import inverse_kinematics, jacobian


EPISODE_LENGTH = 200

tensorboard = SummaryWriter('/run/media/luke/Data/uni/SS2023/DL Lab/Project/qualifying/tqc-4ac' + "/tensorboard/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
# tensorboard = SummaryWriter('.' + "/tensorboard/" + datetime.now().strftime("%Y%m%d-%H%M%S"))


def cust_rewards(env,policy,state,action,next_state,done):
    reward = 0.0
    # done = 0
    ee_pos = policy.get_ee_pose(copy.deepcopy(next_state))[0]                              
    puck_pos = policy.get_puck_pos(copy.deepcopy(next_state))
    # puck_pos = np.array([0.78,0.5,0.1645]) 
    dist = np.linalg.norm(ee_pos[:2]-puck_pos[:2])
    reward += np.exp(-5*dist) * (puck_pos[0]<=1.51)
    mod_next_state = next_state 
    mod_next_state[:3]  = next_state[:3] - [1.51,0,0.1]
    absorbing = env.base_env.is_absorbing(copy.deepcopy(mod_next_state))
    if absorbing:
            # If puck is in the opponent goal
            if (puck_pos[0] - env.base_env.env_info['table']['length'] / 2) > 0 and \
                    (np.abs(puck_pos[1]) - env.base_env.env_info['table']['goal_width']) < 0:
                    print("puck_pos",puck_pos,"absorbing",absorbing)
                    reward = 100
    reward+=policy.get_puck_vel(next_state)[0]
    # # reward -= episode_timesteps*0.01
    # # if policy.get_puck_vel(state)[0]>0.06 and ((dist>0.16)):
    # #     reward+=0
    # reward += np.exp(puck_pos[0]-2.484)*policy.get_puck_vel(state)[0]*(policy.get_puck_vel(state)[0]>0)
    # reward += np.exp(0.536-puck_pos[0])*policy.get_puck_vel(state)[0] *(policy.get_puck_vel(state)[0]<0)
    # reward +=policy.get_puck_vel(next_state)[0]
    des_z = 0.1645
    tolerance = 0.02

    if abs(policy.get_ee_pose(next_state)[0][1])>0.519:
        reward -=10
        done = bool(done+1)

    if (policy.get_ee_pose(next_state)[0][0])<0.536:
        reward -=10
        done = bool(done+1) 
    if (policy.get_ee_pose(next_state)[0][2])<des_z-tolerance*10 or (policy.get_ee_pose(next_state)[0][2])>des_z+tolerance*10:
            reward -=10
            done = bool(done+1)
    reward -= 1e-3 * np.linalg.norm(action)
    # print (reward)

    return reward,done

def _step(env,actor,state,action):
    action = env.action_rescale(action)
    des_pos = np.array([action[0],action[1],0.1645])                                #'ee_desired_height': 0.1645
    _,x = inverse_kinematics(actor.robot_model, actor.robot_data,des_pos)
    x_obs = np.zeros((14))
    x_obs[6:13] = x
    x_ = actor.get_ee_pose(x_obs)[0] 
    y = actor.get_ee_pose(state)[0]
    des_v = action[2]*(x_-y)/np.linalg.norm(x_-y)
    des_v[2] = 0 
    jac = jacobian(actor.robot_model, actor.robot_data,actor.get_joint_pos(state))
    inv_jac = np.linalg.pinv(jac)
    joint_vel = des_v@inv_jac.T[:3,:]
    # if (_):
    action = np.zeros((2,7))
    # pos_max = env.base_env.env_info['robot']['joint_pos_limit'][1]
    # vel_max = env.base_env.env_info['robot']['joint_vel_limit'][1] 
    pos_min = np.array([-0.40476327, -0.20591463, -0.25040716, -1.84715099,\
    -0.28212878, -0.30128033, -0.10303726])
    pos_max = np.array([ 0.16055371, 1.19900776, 0.37962941, -0.92631058,\
    0.3784197, 1.19376235, -0.01559517])
    vel_min = env.env_info['robot']['joint_vel_limit'][0]
    vel_max = env.env_info['robot']['joint_vel_limit'][1]
    min_ = np.stack([pos_min,vel_min]) 
    max_ = np.stack([pos_max,vel_max])
    action[0,:] = x
    action[1:] = joint_vel
    action = action.clip(min_,max_)
    next_state, reward, done, info = env.base_env.step(action)
    reward = reward_mushroomrl(env,actor,copy.deepcopy(next_state),copy.deepcopy(action)) 

    return next_state, reward, done, info


def reward_mushroomrl(env,actor, next_state, action):

    r = 0
    mod_next_state = next_state                            # changing frame of puck pos (wrt origin)
    mod_next_state[:3]  = mod_next_state[:3] - [1.51,0,0.1]
    absorbing = env.base_env.is_absorbing(copy.deepcopy(mod_next_state))
    puck_pos, puck_vel = env.base_env.get_puck(copy.deepcopy(mod_next_state))                     # extracts from obs therefore robot frame


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

    
    ###################################################
    
    

    # If puck is out of bounds
    if absorbing:
        # If puck is in the opponent goal
        if (puck_pos[0] - env.env_info['table']['length'] / 2) > 0 and \
                (np.abs(puck_pos[1]) - env.env_info['table']['goal_width']) < 0:
                print("puck_pos",puck_pos,"absorbing",absorbing)
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
        # dist_ee_board = np.linalg.norm(env.base_env.get_ee()[0][2] -0.0)
        # r -= 1-np.exp(-5*dist_ee_board)
        # print("dist_reward",-1+np.exp(-5*dist_ee_board))
    r -= 1e-3 * np.linalg.norm(action)
    
    des_z = env.base_env.env_info['robot']['ee_desired_height']
    tolerance = 0.02

    if abs(actor.get_ee_pose(copy.deepcopy(next_state))[0][1])>0.519:         # should replace with env variables some day
        r -=0.1 
    if (actor.get_ee_pose(copy.deepcopy(next_state))[0][0])<0.536:
        r -=0.1 
    if (actor.get_ee_pose(copy.deepcopy(next_state))[0][2]-0.1)<des_z-tolerance*10 or (actor.get_ee_pose(copy.deepcopy(next_state))[0][2]-0.1)>des_z+tolerance*10:
        r -=0.1
    return r

class _env(AirHockeyChallengeWrapper):
    def __init__(self, env, custom_reward_function=None, interpolation_order=3, **kwargs):
        super().__init__(env, custom_reward_function, interpolation_order, **kwargs)
        # pos_min = np.array([-0.40476327, -0.20591463, -0.25040716, -1.84715099,\
        #      -0.28212878, -0.30128033, -0.10303726])
        # pos_max = np.array([ 0.16055371, 1.19900776, 0.37962941, -0.92631058,\
        #      0.3784197, 1.19376235, -0.01559517])
        min_ = np.array([0.81,-0.40,-5.0])
        max_ = np.array([1.32,0.40,5.0])
        # vel_min = self.env_info['robot']['joint_vel_limit'][0]
        # vel_max = self.env_info['robot']['joint_vel_limit'][1]
        # min_ = np.stack([pos_min,vel_min]) 
        # max_ = np.stack([pos_max,vel_max])
        # self.max_action  = max_.reshape(14,)
        self.action_space = spaces.Box(low=min_, high=max_,\
             shape=min_.shape, dtype=np.float32)
        self.observation_space = spaces.Box(low=self.env_info['rl_info'].observation_space.low,\
             high=self.env_info['rl_info'].observation_space.high,\
                shape=self.env_info['rl_info'].observation_space.shape, dtype=np.float32)

def main(args, results_dir, models_dir, prefix):
    # --- Init ---

    # remove TimeLimit
    # env = gym.make(args.env).unwrapped
    # eval_env = gym.make(args.env).unwrapped
    env = _env(env="7dof-hit",interpolation_order=3)
    eval_env = _env(env="7dof-hit",interpolation_order=3)

    env = RescaleAction(env, -1., 1.)
    eval_env = RescaleAction(eval_env, -1., 1.)

    state_dim = 23
    action_dim = 3

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    # replay_buffer.load("replay_buffer/data.npz")
    print(replay_buffer.size)
    # replay_buffer = structures.ReplayBuffer(state_dim, action_dim)
    actor = Actor(state_dim, action_dim,env.env_info).to(DEVICE)
    critic = Critic(state_dim, action_dim, args.n_quantiles,
                    args.n_nets).to(DEVICE)
    critic_target = copy.deepcopy(critic)

    top_quantiles_to_drop = args.top_quantiles_to_drop_per_net * args.n_nets

    trainer = Trainer(actor=actor,
                      critic=critic,
                      critic_target=critic_target,
                      top_quantiles_to_drop=top_quantiles_to_drop,
                      discount=args.discount,
                      tau=args.tau,
                      target_entropy=-np.prod(env.action_space.shape).item())
    # trainer.load('/run/media/luke/Data/uni/SS2023/DL Lab/Project/qualifying/tqc-4ac'+'/models/_air-hockey_0')
    evaluations = []
    state, done = env.base_env.reset(), False
    episode_return = 0
    episode_timesteps = 0
    episode_num = 0

    actor.train()
    for t in range(int(args.max_timesteps)):
        action = actor.select_action(state)
        # next_state, reward, done, _ = env.base_env.step(action)
        next_state, reward, done, _ = _step(env,actor,state,action) 
        # reward,done = cust_rewards(env,actor,copy.deepcopy(state),copy.deepcopy(action),\
        #     copy.deepcopy(next_state),copy.deepcopy(done)) 
        # reward = reward_mushroomrl(env,copy.deepcopy(next_state),copy.deepcopy(action))
        # print(reward)
        episode_timesteps += 1
        # env.base_env.render()
        # print(action,reward)
        replay_buffer.add(state, action.reshape(-1,), next_state, reward, done)

        state = next_state
        episode_return += reward

        # Train agent after collecting sufficient data
        if t >= args.batch_size:
            actor_loss,critic_loss,alpha_loss =  trainer.train(replay_buffer, args.batch_size)
            tensorboard.add_scalar("actor loss", actor_loss, t)
            tensorboard.add_scalar("critic loss", critic_loss, t)
            tensorboard.add_scalar("alpha loss", alpha_loss, t)

        if done or episode_timesteps >= EPISODE_LENGTH:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_return:.3f}")
            tensorboard.add_scalar("reward", episode_return, t)
            # Reset environment
            state, done = env.base_env.reset(), False

            episode_return = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            file_name = f"{prefix}_{args.env}_{args.seed}"
            evaluations.append(eval_policy(actor, eval_env, EPISODE_LENGTH))
            np.save(results_dir / file_name, evaluations)
            if args.save_model:
                trainer.save(models_dir / file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # OpenAI gym environment name
    parser.add_argument("--env", default="air-hockey")
    # How often (time steps) we evaluate
    parser.add_argument("--eval_freq", default=2e3, type=int)
    # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n_quantiles", default=25, type=int)
    parser.add_argument("--top_quantiles_to_drop_per_net", default=2, type=int)
    parser.add_argument("--n_nets", default=5, type=int)
    # Batch size for both actor and critic
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--discount", default=0.99,
                        type=float)                 # Discount factor
    # Target network update rate
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument(
        "--log_dir", default='/run/media/luke/Data/uni/SS2023/DL Lab/Project/qualifying/tqc-4ac')
    # parser.add_argument(
    #     "--log_dir", default='.')
    parser.add_argument("--prefix", default='')
    # Save model and optimizer parameters
    parser.add_argument("--save_model", action="store_false")
    args, unknown = parser.parse_known_args()

    log_dir = Path(args.log_dir)

    results_dir = log_dir / 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    models_dir = log_dir / 'models'
    if args.save_model and not os.path.exists(models_dir):
        os.makedirs(models_dir)

    main(args, results_dir, models_dir, args.prefix)
