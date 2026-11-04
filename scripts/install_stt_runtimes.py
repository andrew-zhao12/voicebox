"""Install selected Docker STT runtimes while preserving the main dependency stack."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROFILES = {
    "none": (),
    "faster-whisper": ("faster-whisper",),
    "nemotron": ("nemotron",),
    "all": ("faster-whisper", "nemotron"),
}


def validate_profile(profile: str, variant: str, architecture: str) -> None:
    """Reject unsupported build profiles before expensive dependency installation."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown STT profile {profile!r}; use {', '.join(PROFILES)}")
    if profile != "none" and variant not in ("cpu", "cu128"):
        raise ValueError("Optional STT images support PYTORCH_VARIANT=cpu or cu128 only")
    if profile != "none" and variant == "cu128" and architecture != "amd64":
        raise ValueError("Optional cu128 STT images require linux/amd64")


def install(*, profile: str, variant: str, architecture: str, requirements: Path, destination: Path) -> None:
    """Build the optional environments; reject unsupported combinations before installing anything."""
    validate_profile(profile, variant, architecture)
    destination.mkdir(parents=True, exist_ok=True)
    if "faster-whisper" in PROFILES[profile]:
        # Every transitive dependency is pinned against requirements.lock, so this
        # cannot replace the server's Transformers 4, NumPy, or Hugging Face stack.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--no-deps",
                "-r",
                str(requirements / "requirements-faster-whisper.lock"),
            ],
            check=True,
        )
    if "nemotron" in PROFILES[profile]:
        directory = destination / "nemotron"
        subprocess.run([sys.executable, "-m", "venv", str(directory)], check=True)
        python = str(directory / "bin" / "python")
        version = "2.14.0+cpu" if variant == "cpu" else "2.11.0+cu128"
        # Ignore the builder's pip config: only the chosen wheel index may supply
        # torch here. Installing the remaining locked packages never resolves torch.
        env = {key: value for key, value in os.environ.items() if not key.startswith("PIP_")}
        env["PIP_CONFIG_FILE"] = os.devnull
        subprocess.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--index-url",
                f"https://download.pytorch.org/whl/{variant}",
                f"torch=={version}",
            ],
            check=True,
            env=env,
        )
        subprocess.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--no-deps",
                "--index-url",
                "https://pypi.org/simple",
                "-r",
                str(requirements / "requirements-nemotron.lock"),
            ],
            check=True,
            env=env,
        )
    (destination / "runtimes.json").write_text(
        json.dumps({"profile": profile, "variant": variant, "architecture": architecture}) + "\n"
    )


def main() -> None:
    """Docker build entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--requirements", type=Path, default=Path("/build"))
    parser.add_argument("--destination", type=Path, default=Path("/opt/stt"))
    args = parser.parse_args()
    if args.validate_only:
        validate_profile(args.profile, args.variant, args.architecture)
        return
    install(
        profile=args.profile,
        variant=args.variant,
        architecture=args.architecture,
        requirements=args.requirements,
        destination=args.destination,
    )


if __name__ == "__main__":
    main()
