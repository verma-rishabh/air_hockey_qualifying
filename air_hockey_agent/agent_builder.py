# from air_hockey_agent.ppo import PPO_Agent
from sac_agent import SAC_Agent

def build_agent(env_info,agent_id=1):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    Args:
        env_info (dict): The environment information
        kwargs (any): Additionally setting from agent_config.yml
    Returns:
         (AgentBase) An instance of the Agent
    """

    agent = SAC_Agent(env_info,agent_id)
    # agent = PPO_Agent(env_info,agent_id)
    agent.load("./models/sac_agent/sac")

    return agent    