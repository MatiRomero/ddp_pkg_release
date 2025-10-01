"""CLI wrapper for :mod:`ddp.scripts.plot_shadow_sweep`."""

from __future__ import annotations

from ddp.scripts import plot_shadow_sweep


def main() -> None:
    plot_shadow_sweep.main()


if __name__ == "__main__":  # pragma: no cover - CLI entry-point
    main()
