import copy
import os
import sys
import psutil
import subprocess
import hashlib
import time
from typing import Optional, Tuple, List
import logging
from pathlib import Path

class LaunchWhenAbsent:
    """
    A class to launch a Python script as a detached process if it's not already running.
    If the script is already running, it will skip launching unless force_restart is True.
    """

    def __init__(self, script_path: str, argument_list: List[str] = None, exe: str = None, dir = None, tag='', use_pty=False):
        """
        Initialize with the path to the Python script to be launched.

        Args:
            script_path (str): Full path to the Python script
        """
        self.exe = exe if exe else sys.executable
        self.script_path = os.path.abspath(script_path)
        self.use_pty = use_pty
        if not dir:
            self.dir = os.getcwd()
        else:
            self.dir = dir
        assert os.path.exists(self.dir)
        self.dir = os.path.abspath(self.dir)
        self.argument_list = argument_list

        if not os.path.exists(self.script_path):
            raise FileNotFoundError(f"Script not found: {self.script_path}")

        # Generate unique hash ID for this script

        full_argument_list = [self.script_path] + self.argument_list
        hash_items = full_argument_list + [str(self.dir), str(exe)]
        self.script_hash = hashlib.md5(''.join(hash_items).encode()).hexdigest()[:8]

        # Prepare command with hash ID marker
        if self.use_pty:
            assert len(full_argument_list) == 1
            self.cmd = [self.exe  + " " + full_argument_list[0]]
        else:
            self.cmd = ['nohup'] + [self.exe] + full_argument_list

        log_dir = Path("logs/companion")
        log_dir.mkdir(parents=True, exist_ok=True)
        hostname = os.uname().nodename
        if tag:
            base_log_name = f"{tag}.{self.script_hash}.{hostname}"
        else:
            base_log_name = f"{self.script_hash}.{hostname}"
        self.pgid_file = log_dir / f"{base_log_name}.pgid"
        self.logger_file = log_dir / f"{base_log_name}.log"


    def _is_script_running(self) -> Tuple[bool, Optional[psutil.Process], Optional[int]]:
        """
        Check if the script is already running by looking for its unique hash ID
        in process command lines.

        Returns:
            Tuple[bool, Optional[psutil.Process]]: (is_running, process_if_found)
        """

        # get hostname

        if not self.pgid_file.exists():
            return False, None, None
        else:
            with open(self.pgid_file, 'r') as f_pgid:
                pgid = int(f_pgid.read().strip())
            # Check if the process group ID is still running, if true, psutil
            is_running, proc = self.is_pgid_running(pgid)
            if is_running:
                return True, proc, pgid
            else:
                return False, None, None

    def is_pgid_running(self, pgid):
        for proc in psutil.process_iter(['pid']):
            try:
                if os.getpgid(proc.pid) == pgid:
                    return True, proc
            except (psutil.NoSuchProcess, ProcessLookupError):
                continue
        return False, None

    def _kill_existing_process_group(self, pgid: int):
        """
        Safely terminate the existing process and its children.

        Args:
            pgid (int): Process group ID to terminate
        """
        try:
            # First try SIGTERM for graceful shutdown
            os.killpg(pgid, 15)  # SIGTERM
            print(f"Sent SIGTERM to process group {pgid}")

            # Wait a bit for graceful shutdown
            time.sleep(2)

            os.killpg(pgid, 9)  # SIGKILL
            time.sleep(1)

            print(f"Successfully terminated process group {pgid}")

        except ProcessLookupError:
            # Process group already terminated
            print(f"Process group {pgid} already terminated")
        except PermissionError:
            print(f"Permission denied when trying to kill process group {pgid}")
            raise
        except Exception as e:
            print(f"Error killing process group {pgid}: {e}")
            raise
        finally:
            # Clean up the PGID file
            if self.pgid_file.exists():
                self.pgid_file.unlink()
                print(f"Cleaned up PGID file: {self.pgid_file}")


    def launch(self, force_restart: bool = False, launch_wait_time: int = 30, success_std_string: str = None, env_dict = {}):
        """
        Launch the script if it's not running, or restart it if force_restart is True.

        Args:
            force_restart (bool): If True, kill existing process and restart
            launch_wait_time (int): Maximum time to wait for process launch in seconds
            success_std_string (str): String to look for in stdout to confirm successful launch
        """
        is_running, existing_process, pgid = self._is_script_running()


        if is_running:
            if force_restart:
                print(f"Force restarting")
                self._kill_existing_process_group(pgid)
            else:
                print(f"Script is already running, skipping launch")
                return
        try:
            # Set up process creation flags and environment
            # Create logs directory
            log_dir = Path("logs/companion")
            log_dir.mkdir(parents=True, exist_ok=True)

            # Open log file
            log_file = self.logger_file

            if os.name == 'nt':  # Windows
                # DETACHED_PROCESS flag
                raise NotImplementedError("Windows support is not implemented yet.")
            else:  # Unix-like systems
                # Use nohup and redirect output
                print("log to", log_file)
                print("launching", " ".join(self.cmd))
                # Open log file
                if log_file.exists():
                    os.remove(log_file)
                if not self.use_pty:
                    f = open(log_file, 'a')
                    proc = subprocess.Popen(
                        self.cmd,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        cwd=self.dir,
                        env={'ScriptHash': self.script_hash,**os.environ, **env_dict},
                        start_new_session=True  # Start new session
                    )
                    f.close()  # Close append handle
                    pgid = os.getpgid(proc.pid)
                else:
                    import base64

                    def string_to_base64(s):
                        # encode into bytes
                        s_bytes = s.encode('utf-8')
                        # to base64
                        base64_bytes = base64.b64encode(s_bytes)
                        # to string
                        base64_string = base64_bytes.decode('utf-8')
                        return base64_string

                    f = open(log_file, 'a')
                    converted_cmd = [
                            sys.executable,
                            "-m",
                            "agentevolver.utils.pty",
                            "--human-cmd", f"'{string_to_base64(self.cmd[0])}'",
                            "--dir", self.dir,
                            "--env", str(env_dict),
                        ]
                    print('running pty command:', ' '.join(converted_cmd))
                    proc = subprocess.Popen(
                        converted_cmd,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        cwd="./",
                        env={'ScriptHash': self.script_hash,**os.environ, **env_dict},
                        start_new_session=True  # Start new session
                    )
                    f.close()  # Close append handle
                    pgid = os.getpgid(proc.pid)

                # write pgid to {log_file}.pgid
                with open(self.pgid_file, 'w') as f_pgid:
                    f_pgid.write(str(pgid))

                # Monitor log file for success string or timeout
                start_time = time.time()
                f_read = ""
                previous_r_print = False
                with open(log_file, 'r') as f:
                    while time.time() - start_time < launch_wait_time:
                        f_read_ = f.read()
                        inc_read = f_read_[len(f_read):]
                        f_read = f_read_  # Update f_read to the latest content
                        if success_std_string:
                            # Move to end of file and read new content
                            if success_std_string in f_read:
                                print(f"Found success string '{success_std_string}' in output")
                                break
                        time.sleep(1)
                        remaining = int(launch_wait_time - (time.time() - start_time))
                        f_read_trim = inc_read.replace('\n', ' ')
                        if f_read_trim:
                            if previous_r_print: print('')
                            print(f"Waiting for process launch... {remaining}s remaining ({f_read_trim})")
                            previous_r_print = False
                        else:
                            print(f"\rWaiting for process launch... {remaining}s remaining", end='', flush=True)
                            previous_r_print = True

                        if remaining % 10 == 0:
                            is_running, proc = self.is_pgid_running(pgid)
                            if not is_running:
                                raise RuntimeError(f"Process with PGID {pgid} is not running, cannot confirm launch")

                    else:
                        if success_std_string:
                            raise TimeoutError(f"Process did not output success string '{success_std_string}' within {launch_wait_time} seconds")

                print(f"Successfully launched {self.cmd} with PID {proc.pid}")

        except Exception as e:
            logging.error(f"Error launching script: {e}")
            raise


class LaunchCommandWhenAbsent(LaunchWhenAbsent):
    def __init__(self, full_argument_list: List[str], dir = None, tag = "", use_pty=False):
        if not dir:
            self.dir = os.getcwd()
        else:
            self.dir = dir
        assert os.path.exists(self.dir)
        self.dir = os.path.abspath(self.dir)
        self.use_pty = use_pty

        full_argument_list_compute_hash = full_argument_list.copy()
        if full_argument_list_compute_hash[0] == sys.executable:
            full_argument_list_compute_hash[0] = 'python'

        hash_items = full_argument_list_compute_hash + [str(self.dir)]
        self.script_hash = hashlib.md5(''.join(hash_items).encode()).hexdigest()[:8]
        if self.use_pty:
            assert len(full_argument_list) == 1
            self.cmd = full_argument_list
        else:
            self.cmd = ['nohup'] + full_argument_list
        # raise ValueError(self.script_hash)

        log_dir = Path("logs/companion")
        log_dir.mkdir(parents=True, exist_ok=True)
        hostname = os.uname().nodename
        if tag:
            base_log_name = f"{tag}.{self.script_hash}.{hostname}"
        else:
            base_log_name = f"{self.script_hash}.{hostname}"
        self.pgid_file = log_dir / f"{base_log_name}.pgid"
        self.logger_file = log_dir / f"{base_log_name}.log"
