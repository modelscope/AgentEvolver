import os

"""
Ray Distributed Debugger VSCode Extension (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Starting with Ray 2.39, Anyscale has introduced the `Ray Distributed Debugger <https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html>`_ VSCode extension. Follow the extension’s installation instructions, then add your cluster using the dashboard URL you obtained earlier.

   .. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/debugger.png?raw=true
      :alt: Ray Distributed Debugger VSCode extension screenshot

2. Prerequisites.

   Ensure the following are installed (see the extension README for more detail):

   - Visual Studio Code
   - `ray[default]` >= 2.9.1
   - `debugpy` >= 1.8.0

   .. image:: https://github.com/aoshen524/verl/blob/main/docs/start/c7098b755ff689859837773a916c857.png?raw=true
      :alt: VSCode with Ray prerequisites

3. Environment Variables.

   To enable post‑mortem debugging, set:

   .. code-block:: bash

      export RAY_DEBUG_POST_MORTEM=1

   .. admonition:: Note
      :class: important

      Be sure to remove any legacy flags before starting Ray:

      - `RAY_DEBUG=legacy`
      - `--ray-debugger-external`

4. Configuring BreakpointsSet up breakpoint() in your code, and submit job to cluster. Then the extension will show the breakpoint information.


   1. Insert `breakpoint()` calls into your remote functions.
   2. Submit your job to the cluster.

   The extension will detect active breakpoints and display them in VSCode.

   .. image:: https://github.com/aoshen524/verl/blob/main/docs/start/4ddad74395c79a1402331c0ce73316f.png?raw=true
      :alt: Detected breakpoint in VSCode

   **Note:** Breakpoints are only supported inside functions decorated with `@ray.remote`.

5. Launching the Debugger.

   Run your job directly from the command line (do not use a `launch.json`):

   .. code-block:: bash

      python job.py

6. Attaching to a Breakpoint.

 Once the process hits the first `breakpoint()`, click the Ray Distributed Debugger icon in the VSCode sidebar to attach the debugger.

   .. image:: https://github.com/aoshen524/verl/blob/main/docs/start/4ddad74395c79a1402331c0ce73316f.png?raw=true
      :alt: Attaching VSCode debugger to Ray process

7. Debugging With Multiple breakpoint().

   For each subsequent task, first disconnect the current debugger session, then click the extension icon again to attach to the next breakpoint.

   .. image:: https://github.com/aoshen524/verl/blob/main/docs/start/6e83c910a62c82fecb89c6619e001cd.png?raw=true
      :alt: Disconnecting and reconnecting the debugger
"""

def vscode_conditional_breakpoint(tag=None, rank=-1, once=True):
   """
   Set a conditional breakpoint hook for VS Code–style debugging under Ray.

   Triggers ``breakpoint()`` when all of the following hold:
   1. ``rank`` is 0 or the ``RANK`` env is 0 (optional gate; see implementation).
   2. ``RAY_DEBUG_POST_MORTEM`` is set in the environment.
   3. If ``tag`` is given, it must appear in ``DEBUG_TAGS`` (pipe-separated list).

   Args:
      tag: Optional label matched against ``DEBUG_TAGS``. If ``None``, behavior depends on ``once``.
      rank: Process/GPU rank; used to restrict which worker hits the breakpoint.

   Env vars:
      RAY_DEBUG_POST_MORTEM: If unset, the function returns immediately.
      DEBUG_TAGS: Pipe-separated tags; ``tag`` must be one of them when ``tag`` is not ``None``.
   """
   env_tag = f'HIT_BREAKPOINT_REC_{tag}'
   # if rank < 0: rank = os.getenv("RANK", 0)
   # if rank != 0: return
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