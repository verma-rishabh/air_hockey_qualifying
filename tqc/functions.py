import torch
import numpy as np
from air_hockey_challenge.utils.kinematics import inverse_kinematics, jacobian
from tqc import DEVICE
import copy


def _step(env,actor,state,action):
    action = env.action_rescale(action)
    des_pos = np.array([action[0],action[1],0.1645])                                #'ee_desired_height': 0.1645
    _,x = inverse_kinematics(actor.robot_model, actor.robot_data,des_pos)
    x_obs = np.zeros((14))
    x_obs[6:13] = x
    x_ = actor.get_ee_pose(x_obs)[0] 
    y = actor.get_ee_pose(state)[0]
    des_v = action[2]*(x_-y)/np.linalg.norm(x_-y) 
    jac = jacobian(actor.robot_model, actor.robot_data,actor.get_joint_pos(state))
    inv_jac = np.linalg.pinv(jac)
    joint_vel = des_v@inv_jac.T[:3,:]
    # if (_):
    action = np.zeros((2,7))
    pos_max = env.base_env.env_info['robot']['joint_pos_limit'][1]
    vel_max = env.base_env.env_info['robot']['joint_vel_limit'][1] 
    max_ = np.stack([pos_max,vel_max])


    action[0,:] = x
    action[1:] = joint_vel
    action = action.clip(-max_,max_)
    next_state, reward, done, info = env.base_env.step(action)
    reward = reward_mushroomrl(env,copy.deepcopy(next_state),copy.deepcopy(action)) 

    return next_state, reward, done, info
# def cust_rewards(env,policy,state,action,next_state,done):
#     reward = 0.0
#     # done = 0
#     ee_pos = policy.get_ee_pose(copy.deepcopy(next_state))[0]                              
#     puck_pos = policy.get_puck_pos(copy.deepcopy(next_state))
#     # puck_pos = np.array([0.78,0.5,0.1645]) 
#     dist = np.linalg.norm(ee_pos[:2]-puck_pos[:2])
#     reward += np.exp(-5*dist) * (puck_pos[0]<=1.51)
#     mod_next_state = next_state 
#     mod_next_state[:3]  = next_state[:3] - [1.51,0,0.1]
#     absorbing = env.base_env.is_absorbing(copy.deepcopy(mod_next_state))
#     if absorbing:
#             # If puck is in the opponent goal
#             if (puck_pos[0] - env.base_env.env_info['table']['length'] / 2) > 0 and \
#                     (np.abs(puck_pos[1]) - env.base_env.env_info['table']['goal_width']) < 0:
#                     print("puck_pos",puck_pos,"absorbing",absorbing)
#                     reward = 100
#     reward+=policy.get_puck_vel(next_state)[0]
#     # # reward -= episode_timesteps*0.01
#     # # if policy.get_puck_vel(state)[0]>0.06 and ((dist>0.16)):
#     # #     reward+=0
#     # reward += np.exp(puck_pos[0]-2.484)*policy.get_puck_vel(state)[0]*(policy.get_puck_vel(state)[0]>0)
#     # reward += np.exp(0.536-puck_pos[0])*policy.get_puck_vel(state)[0] *(policy.get_puck_vel(state)[0]<0)
#     # reward +=policy.get_puck_vel(next_state)[0]
#     des_z = 0.1645
#     tolerance = 0.02

#     if abs(policy.get_ee_pose(next_state)[0][1])>0.519:
#         reward -=10
#         # done = 1

#     if (policy.get_ee_pose(next_state)[0][0])<0.536:
#         reward -=10
#         # done = 1 
#     if (policy.get_ee_pose(next_state)[0][2])<des_z-tolerance*10 or (policy.get_ee_pose(next_state)[0][2])>des_z+tolerance*10:
#             reward -=10
#             # done = 1
#     reward -= 1e-3 * np.linalg.norm(action)
#     # print (reward)

#     return reward,done

def _step(env,actor,state,action):
    action = env.action_rescale(action)
    des_pos = np.array([action[0],action[1],0.1645])                                #'ee_desired_height': 0.1645
    _,x = inverse_kinematics(actor.robot_model, actor.robot_data,des_pos)
    x_obs = np.zeros((14))
    x_obs[6:13] = x
    x_ = actor.get_ee_pose(x_obs)[0] 
    y = actor.get_ee_pose(state)[0]
    des_v = action[2]*(x_-y)/np.linalg.norm(x_-y) 
    jac = jacobian(actor.robot_model, actor.robot_data,actor.get_joint_pos(state))
    inv_jac = np.linalg.pinv(jac)
    joint_vel = des_v@inv_jac.T[:3,:]
    # if (_):
    action = np.zeros((2,7))
    pos_max = env.base_env.env_info['robot']['joint_pos_limit'][1]
    vel_max = env.base_env.env_info['robot']['joint_vel_limit'][1] 
    max_ = np.stack([pos_max,vel_max])


    action[0,:] = x
    action[1:] = joint_vel
    action = action.clip(-max_,max_)
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
        r -=1 
    if (actor.get_ee_pose(copy.deepcopy(next_state))[0][0])<0.536:
        r -=1 
    if (actor.get_ee_pose(copy.deepcopy(next_state))[0][2]-0.1)<des_z-tolerance*10 or (actor.get_ee_pose(copy.deepcopy(next_state))[0][2]-0.1)>des_z+tolerance*10:
        r -=1
    return r
def eval_policy(policy, eval_env, max_episode_steps, eval_episodes=7):
    policy.eval()
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        t = 0
        while not done and t < max_episode_steps:
            action = policy.select_action(state)
            # next_state, reward, done, _= eval_env.step(action)
            next_state, reward, done, _ = _step(eval_env,policy,state,action) 
            # reward,done = cust_rewards(eval_env,policy,state,action,next_state,done)
            # reward = reward_mushroomrl(eval_env,copy.deepcopy(next_state),copy.deepcopy(action)) 
            print(action,reward)
            avg_reward += reward
            t += 1
            eval_env.base_env.render()
            state = next_state
    avg_reward /= eval_episodes
    policy.train()
    return avg_reward


def quantile_huber_loss_f(quantiles, samples):
    pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1,
                             abs_pairwise_delta - 0.5,
                             pairwise_delta ** 2 * 0.5)

    n_quantiles = quantiles.shape[2]
    tau = torch.arange(n_quantiles, device=DEVICE).float() / n_quantiles + 1 / 2 / n_quantiles
    loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
    return loss
