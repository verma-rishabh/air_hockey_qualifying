import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from air_hockey_challenge.framework.agent_base import AgentBase
from omegaconf import OmegaConf
from utils import solve_hit_config_ik_null

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_dim)
        
        # self.max_action = max_action
        

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 128)
        self.l5 = nn.Linear(128, 128)
        self.l6 = nn.Linear(128, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3_agent(AgentBase):
    def __init__(
        self,
        env_info,
        agent_id,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
    ):
        super().__init__(env_info, agent_id)
        conf = OmegaConf.load('train_td3.yaml')

        state_dim = env_info["rl_info"].observation_space.shape[0]
        #action_dim = env_info["rl_info"].action_space.shape[0]
        action_dim = 3
        #pos_max = env_info['robot']['joint_pos_limit'][1]
        #vel_max = env_info['robot']['joint_vel_limit'][1] 
        #max_ = np.stack([pos_max,vel_max],dtype=np.float32)
        #max_action  = max_.reshape(14,)
        self.min_action = torch.from_numpy(np.array([0.65,-0.40,0],dtype=np.float32)).to(device)
        self.max_action = torch.from_numpy(np.array([1.32,0.40,1.5],dtype=np.float32)).to(device)
        state_max = np.array(env_info['rl_info'].observation_space.high,dtype=np.float32)
        self.state_max = torch.from_numpy(state_max).to(device)
        # max_action = np.array([1.5,0.5,5],dtype=np.float32)
        # max_action = torch.from_numpy(max_action).to(device)
        discount = conf.agent.discount
        tau=conf.agent.tau
        policy_noise=conf.agent.policy_noise
        noise_clip=conf.agent.noise_clip
        policy_freq=conf.agent.policy_freq
        
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        # self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state)[0].cpu().data.numpy().flatten()

    def action_scaleup(self, action):
        low = self.min_action.detach().cpu().numpy()
        high = self.max_action.detach().cpu().numpy()
        a = np.zeros_like(low) -1.0
        b = np.zeros_like(low) +1.0
        action = low + (high - low)*((action - a)/(b - a))
        action = np.clip(action, low, high)
        return action

    def draw_action(self, state):
        norm_state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.action_scaleup(self.actor(norm_state)[0].detach().cpu().numpy())
        des_pos = np.array([action[0],action[1],0.1645])                                #'ee_desired_height': 0.1645
        x_ = [action[0],action[1]] 
        y = self.get_ee_pose(state)[0][:2]
        des_v = action[2]*(x_-y)/(np.linalg.norm(x_-y)+1e-8)
        des_v = np.concatenate((des_v,[0])) 
        # _,x = inverse_kinematics(self.policy.robot_model, self.policy.robot_data,des_pos)
        _,x = solve_hit_config_ik_null(self.robot_model,self.robot_data, des_pos, des_v, self.get_joint_pos(state))
        return x


    def train(self, replay_buffer, batch_size=1024):
        _actor_loss = np.nan
        _critic_loss = np.nan

        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(self.min_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        _critic_loss = critic_loss.item()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            _actor_loss = actor_loss.item()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return _actor_loss, _critic_loss

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
        