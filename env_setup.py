"""
env_setup.py
============
Loads ./.env and reports which sponsor credentials are configured.
Run it any time to check your setup:

    python env_setup.py

Other modules can do `from env_setup import load_keys; load_keys()` to make
sure the .env values are present in os.environ before using a sponsor SDK.
"""

import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent / ".env"


def load_keys():
    """Load .env into os.environ (uses python-dotenv if available, else a
    tiny built-in parser so it works even before you pip-install dotenv)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_PATH)
        return
    except ImportError:
        pass
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


# name -> (what it's for, is it required to win)
SPONSORS = [
    ("HUD_API_KEY", "HUD - RL environment + eval platform (hud.ai)", True),
    ("ANTHROPIC_API_KEY", "Anthropic - Claude coordinator agent", True),
    ("FIREWORKS_API_KEY", "Fireworks - fine-tuned open models (RFT)", False),
    ("MODAL_TOKEN_ID", "Modal - compute (or run `modal token new`)", False),
    ("ANTIM_API_KEY", "Antim Labs - physical-AI sim (confirm name)", False),
]


def main():
    load_keys()
    print("Sponsor credential check  (.env)\n" + "-" * 52)
    for key, desc, required in SPONSORS:
        ok = bool(os.environ.get(key))
        mark = "OK " if ok else ("MISSING" if required else "not set")
        tag = "[required]" if required else "[optional]"
        print(f"  {mark:8} {key:20} {tag}  {desc}")
    # Modal can also be authed via its own token file (modal token new)
    modal_file = Path.home() / ".modal.toml"
    print(f"\n  Modal token file (~/.modal.toml): "
          f"{'found' if modal_file.exists() else 'not found - run: modal token new'}")
    print("-" * 52)


if __name__ == "__main__":
    main()
