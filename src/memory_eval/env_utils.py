"""Environment variable loading.

The .env file is NOT expected inside this repository. It is looked up, in order:
  1. the explicit `env_file` argument (e.g. the --env-file command line option),
  2. the MEMPOISON_ENV_FILE environment variable,
  3. the nearest .env found walking up from the current working directory
     (so running from the parent repository picks up the parent's .env).

`API_KEY` and `Base_URL` stay available as module attributes for the other eval scripts
(`env_utils.API_KEY`, `from env_utils import API_KEY`). They are resolved lazily, so merely
importing this module does not read any file.
"""
import os

from dotenv import find_dotenv, load_dotenv

_loaded_path = None
_loaded = False


def load_env(env_file=None, override=True):
    """Load a .env file into os.environ. Returns the path loaded, or None if none was found."""
    global _loaded, _loaded_path
    path = env_file or os.environ.get("MEMPOISON_ENV_FILE") or find_dotenv(usecwd=True)
    if env_file and not os.path.isfile(env_file):
        raise FileNotFoundError(f"env file not found: {env_file}")
    if path:
        load_dotenv(path, override=override)
    _loaded, _loaded_path = True, (path or None)
    return _loaded_path


def __getattr__(name):
    if name in ("API_KEY", "Base_URL"):
        if not _loaded:
            load_env()
        return os.getenv(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
