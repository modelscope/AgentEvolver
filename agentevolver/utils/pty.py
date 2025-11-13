import os
import pty

def run_command_with_pty(cmd, working_dir, env_dict):
    """
    Run a command using a pseudo-terminal and write the output to a log file.

    Parameters:
        cmd (list): The command to run (e.g. ["ls", "-l"]).
        working_dir (str): Working directory.
        env_dict (dict): Environment variables dictionary.
    """
    # Save original environment variables
    original_env = os.environ.copy()
    original_dir = os.getcwd()

    try:
        # Switch to the specified working directory
        os.chdir(working_dir)

        # Update environment variables
        for key, value in env_dict.items():
            os.environ[key] = value

        # # Open log file in append mode to write
        # with open(log_file, 'a') as log_f:

        # Define master device read callback function
        def master_read(fd):
            try:
                # Read data from the master device
                data = os.read(fd, 1024)
            except OSError:
                return b""

            if data:
                # Write data to log file
                # log_f.write(data.decode())
                # log_f.flush()
                # Also print to standard output (optional)
                print(data.decode(), end="")
            return data

        # Define standard input read callback function
        def stdin_read(fd):
            # If no data needs to be read from standard input, return empty bytes directly
            return b""

        # Use pty.spawn to allocate a pseudo-terminal and run the command
        pty.spawn(cmd, master_read, stdin_read)

    finally:
        # Restore original working directory
        os.chdir(original_dir)

        # Restore original environment variables
        os.environ.clear()
        os.environ.update(original_env)

import base64

# Convert string to Base64
def string_to_base64(s):
    # First encode the string to bytes
    s_bytes = s.encode('utf-8')
    # Convert bytes to base64
    base64_bytes = base64.b64encode(s_bytes)
    # Convert base64 bytes back to string
    base64_string = base64_bytes.decode('utf-8')
    return base64_string

# Convert Base64 back to string
def base64_to_string(b):
    # Convert base64 string to bytes
    base64_bytes = b.encode('utf-8')
    # Decode base64 bytes
    message_bytes = base64.b64decode(base64_bytes)
    # Convert bytes back to string
    message = message_bytes.decode('utf-8')
    return message

def pty_wrapper(
    cmd: list[str],
    dir: str,
    env_dict: dict = {},
):
    run_command_with_pty(cmd, working_dir=dir, env_dict=env_dict)

def pty_wrapper_final(human_cmd, dir, env_dict):
    pty_wrapper(["/bin/bash", "-c", human_cmd], dir, env_dict)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run a shell command in a PTY with logging and custom env.")
    parser.add_argument("--human-cmd", type=str, help="Shell command to run (as a string)")
    parser.add_argument("--dir", type=str, default=".", help="Working directory")
    parser.add_argument("--env", type=str, default="{}", help="Environment variables as JSON string, e.g. '{\"KEY\":\"VAL\"}'")

    args = parser.parse_args()

    try:
        env_dict = json.loads(args.env)
        if not isinstance(env_dict, dict):
            raise ValueError
    except Exception:
        print("--env must be a valid JSON object string, e.g. '{\"KEY\":\"VAL\"}'")
        exit(1)

    pty_wrapper_final(base64_to_string(args.human_cmd), args.dir, env_dict)

