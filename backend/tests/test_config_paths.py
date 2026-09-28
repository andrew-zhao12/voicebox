"""Storage paths relative to the data directory, including data dirs that contain a ``data`` component (torch-free)."""

from backend import config


def test_storage_paths_survive_a_data_dir_with_a_data_component(tmp_path):
    data_dir = tmp_path / "data" / "voicebox"
    previous = config.get_data_dir()
    config.set_data_dir(data_dir)
    try:
        sample = data_dir / "profiles" / "p1" / "s1.wav"
        sample.parent.mkdir(parents=True)
        sample.write_bytes(b"RIFF")

        stored = config.to_storage_path(sample)
        assert stored == "profiles/p1/s1.wav"
        assert config.resolve_storage_path(stored) == sample.resolve()
        # Absolute paths inside the data dir resolve to themselves.
        assert config.resolve_storage_path(str(sample.resolve())) == sample.resolve()
        # Legacy records: a relative path with the data-dir name baked in, and an
        # absolute path from another machine, still land inside this data dir.
        assert config.resolve_storage_path("data/profiles/p1/s1.wav") == sample.resolve()
        assert config.resolve_storage_path("/srv/old/data/profiles/p1/s1.wav") == sample.resolve()
        # Files outside every data dir stay absolute.
        outside = tmp_path / "elsewhere.wav"
        assert config.to_storage_path(outside) == str(outside.resolve())
        assert config.resolve_storage_path("") is None
    finally:
        config.set_data_dir(previous)
