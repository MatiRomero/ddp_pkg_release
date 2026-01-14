"""Logging utilities for experiment runs."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_path: Path) -> logging.Logger:
    """Set up logging to both console and file.
    
    Creates a logger that writes to both the console and a file. Uses INFO level
    and timestamps each line. Avoids duplicate handlers if called multiple times.
    
    Parameters
    ----------
    log_path : Path
        Path to the log file. Parent directory will be created if needed.
        
    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger("ddp.run")
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    # Create formatter with timestamp
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

