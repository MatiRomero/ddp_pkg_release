"""Repository wrapper for ``ddp.scripts.meituan_shadow_sweep``."""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    """Invoke the package CLI with the provided arguments."""

    command = [sys.executable, "-m", "ddp.scripts.meituan_shadow_sweep", *sys.argv[1:]]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

