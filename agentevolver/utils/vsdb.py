# Copyright 2025 Ablibaba Ltd. and/or its affiliates


import os
import pickle

"""
This debug util works together with the Ray Distributed Debugger VSCode Extension.
For more details, please refer to:
   https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html

For usage together with AgentEvolver launcher, please refer to:
   docs/launcher.md
"""

def vscode_conditional_breakpoint(tag=None, once=True):
   """
   Conditionally set a breakpoint for VSCode debugging with Ray distributed systems.

   This function provides a smart breakpoint mechanism that respects environment
   variables and can be configured to trigger only once or multiple times based on
   debug tags. It's designed to work with Ray's post-mortem debugging feature.

   Args:
       tag (str, optional): A debug tag to conditionally trigger the breakpoint.
           If provided, the breakpoint will only trigger if this tag is present
           in the DEBUG_TAGS environment variable (pipe-separated list).
           If None, the breakpoint behavior depends only on the `once` parameter.
           Defaults to None.

       once (bool, optional): Whether the breakpoint should only trigger once
           per tag/session. If True, uses environment variables to track if
           the breakpoint has already been hit. If False, the breakpoint will
           trigger every time the function is called (subject to other conditions).
           Defaults to True.

   Returns:
       None: This function doesn't return any value. It either triggers a
       breakpoint or returns silently.

   Environment Variables:
       RAY_DEBUG_POST_MORTEM: Must be set to enable any breakpoint functionality.
           If not set, the function returns immediately without doing anything.

       DEBUG_TAGS: Pipe-separated list of debug tags (e.g., "tag1|tag2|tag3").
           Only required when using the `tag` parameter. The breakpoint will
           only trigger if the provided tag is found in this list.

       HIT_BREAKPOINT_REC_{tag}: Automatically created environment variables
           to track whether a specific tagged breakpoint has already been hit
           when `once=True`. These are internal tracking variables.

   Examples:
       # Simple breakpoint that triggers once
       vscode_conditional_breakpoint(tag="training")

       # Breakpoint that triggers every time
       vscode_conditional_breakpoint(tag="training", once=False)

       # Tagged breakpoint (requires DEBUG_TAGS="training|validation")
       vscode_conditional_breakpoint(tag="training")

       # Tagged breakpoint that can trigger multiple times
       vscode_conditional_breakpoint(tag="validation", once=False)

   Note:
       This function is designed to work with Ray's distributed debugging
       capabilities and the VSCode Ray Distributed Debugger extension.
       Make sure RAY_DEBUG_POST_MORTEM=1 is set in your environment.
   """

   env_tag = f'HIT_BREAKPOINT_REC_{tag}'
   if not os.getenv('RAY_DEBUG_POST_MORTEM'): return
   if tag is None:
      if once:
         if os.getenv(env_tag, "") != "1":
            os.environ[env_tag] = "1"
            breakpoint()
            return
      else:
         breakpoint()
         return
   else:
      debug_tags = os.getenv('DEBUG_TAGS', '').split('|')
      if tag in debug_tags:
         if once:
            if os.getenv(env_tag, "") != "1":
               os.environ[env_tag] = "1"
               breakpoint()
               return
         else:
            breakpoint()
            return


def objdump(obj, file="objdump.tmp"):
   with open(file, "wb+") as f:
      pickle.dump(obj, f)
   return


def objload(file="objdump.tmp"):
   import os
   if not os.path.exists(file):
      return
   with open(file, "rb") as f:
      return pickle.load(f)


bp = vscode_conditional_breakpoint
