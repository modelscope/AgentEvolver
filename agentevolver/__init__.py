# Define the current version of the project
__version__ = "0.1.0"  # ⭐ Sets the version of the project


def get_version():
    """Return the AgentEvolver version."""
    return __version__


from agentevolver.utils.vsdb import vscode_conditional_breakpoint as bp