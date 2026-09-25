"""Tests for ``python -m backend.keys``."""

import pytest

from backend import config
from backend.keys import main


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("VOICEBOX_API_KEY", raising=False)
    monkeypatch.delenv("VOICEBOX_API_KEY_FILE", raising=False)
    monkeypatch.delenv("VOICEBOX_API_KEYS_JSON", raising=False)
    previous = config.get_data_dir()
    yield tmp_path
    config.set_data_dir(previous)


def test_create_list_revoke_and_local(data_dir, capsys):
    assert (
        main(["--data-dir", str(data_dir), "create", "--id", "app", "--role", "client", "--limit", "requests=5"]) == 0
    )
    key = capsys.readouterr().out.strip()
    assert key.startswith("vbx_")

    main(["--data-dir", str(data_dir), "list"])
    out = capsys.readouterr().out
    assert "app" in out
    assert "requests=5" in out
    assert key not in out

    main(["--data-dir", str(data_dir), "local"])
    local = capsys.readouterr().out.strip()
    assert local == (data_dir / "api_key").read_text().strip()

    main(["--data-dir", str(data_dir), "path"])
    assert str(data_dir / "api_keys.json") in capsys.readouterr().out

    assert main(["--data-dir", str(data_dir), "revoke", "--id", "app"]) == 0
    with pytest.raises(SystemExit):
        main(["--data-dir", str(data_dir), "revoke", "--id", "app"])


def test_bad_inputs_exit_nonzero(data_dir):
    with pytest.raises(SystemExit):
        main(["--data-dir", str(data_dir), "create", "--id", "env"])
    with pytest.raises(SystemExit):
        main(["--data-dir", str(data_dir), "create", "--id", "ok", "--limit", "requests"])
    with pytest.raises(SystemExit):
        main(["--data-dir", str(data_dir), "create", "--id", "ok", "--limit", "bogus=1"])
