from agent import Agent


def default_policy(agent: Agent) -> str:
    """
    Policy of the agent
    return "left", "right", or "none"
    """
    actions = ["left", "right", "none"]
    action = "left"
    assert action in actions
    return action
