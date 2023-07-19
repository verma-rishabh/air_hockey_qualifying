from air_hockey_agent.TQC_agent import TQC_agent 


def build_agent(env_info, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    Args:
        env_info (dict): The environment information
        kwargs (any): Additionally setting from agent_config.yml
    Returns:
         (AgentBase) An instance of the Agent
    """

    agent = TQC_agent(env_info,agent_id=2)
    agent.load("/home/captain/SS23/DeepLearning Lab/Project/air_hockey_qualifying/tournament/models/tqc-hit-mushroomrl_7dof-hit_0")
    return agent
