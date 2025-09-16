#!/usr/bin/env python3
"""Backward-compatible entry point that delegates to :mod:`sweep_param`."""

from __future__ import annotations

from ddp.scripts.sweep_param import main


if __name__ == "__main__":
    main()

