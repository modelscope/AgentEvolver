import pytest

import launcher


def test_pty_launch_requires_reme_env_vars(monkeypatch):
    monkeypatch.delenv("REME_PATH", raising=False)
    monkeypatch.delenv("REME_SCRIPT", raising=False)

    with pytest.raises(RuntimeError, match="REME_PATH, REME_SCRIPT"):
        launcher.pty_launch("reme")


def test_pty_launch_requires_single_missing_env_var(monkeypatch):
    monkeypatch.setenv("REME_PATH", "/tmp/reme")
    monkeypatch.delenv("REME_SCRIPT", raising=False)

    with pytest.raises(RuntimeError, match="REME_SCRIPT"):
        launcher.pty_launch("reme")
