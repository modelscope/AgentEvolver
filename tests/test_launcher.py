from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import launcher


class DummyCompanion:
    def __init__(self, full_argument_list, dir, tag, use_pty):
        self.full_argument_list = full_argument_list
        self.dir = dir
        self.tag = tag
        self.use_pty = use_pty
        self.launch_calls = []

    def launch(self, launch_wait_time, success_std_string):
        self.launch_calls.append(
            {
                "launch_wait_time": launch_wait_time,
                "success_std_string": success_std_string,
            }
        )



def test_require_service_env_rejects_missing_vars(monkeypatch):
    monkeypatch.delenv("REME_PATH", raising=False)
    monkeypatch.delenv("REME_SCRIPT", raising=False)

    with pytest.raises(ValueError, match="REME_PATH, REME_SCRIPT"):
        launcher._require_service_env("reme")



def test_pty_launch_uses_validated_service_env(monkeypatch, tmp_path):
    monkeypatch.setenv("REME_PATH", str(tmp_path))
    monkeypatch.setenv("REME_SCRIPT", "python -m reme.api")

    created = {}

    def fake_launch_command_when_absent(full_argument_list, dir, tag, use_pty):
        companion = DummyCompanion(full_argument_list, dir, tag, use_pty)
        created["companion"] = companion
        return companion

    monkeypatch.setattr(launcher, "LaunchCommandWhenAbsent", fake_launch_command_when_absent)

    launcher.pty_launch("reme", success_std_string="Uvicorn running on")

    companion = created["companion"]
    assert companion.full_argument_list == ["python -m reme.api"]
    assert companion.dir == str(tmp_path)
    assert companion.tag == "appworld_env_service"
    assert companion.use_pty is True
    assert companion.launch_calls == [
        {
            "launch_wait_time": 1800,
            "success_std_string": "Uvicorn running on",
        }
    ]
