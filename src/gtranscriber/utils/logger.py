"""Rich logging integration for G-Transcriber.

Provides structured logging with Rich console output.
"""

from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler

# Global console instances
console = Console()
stderr_console = Console(stderr=True)


def setup_logging(
    level: int = logging.INFO,
    show_time: bool = True,
    show_path: bool = False,
) -> logging.Logger:
    """Configure logging with Rich handler.

    Args:
        level: Logging level.
        show_time: Whether to show timestamps.
        show_path: Whether to show file paths.

    Returns:
        Configured logger instance.
    """
    # Create Rich handler
    handler = RichHandler(
        console=console,
        show_time=show_time,
        show_path=show_path,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
    )

    # Configure format
    handler.setFormatter(logging.Formatter("%(message)s"))

    # Get the root logger
    logger = logging.getLogger("gtranscriber")
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Add Rich handler
    logger.addHandler(handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (optional).

    Returns:
        Logger instance.
    """
    if name:
        return logging.getLogger(f"gtranscriber.{name}")
    return logging.getLogger("gtranscriber")


def log_info(message: str) -> None:
    """Log an info message.

    Args:
        message: Message to log.
    """
    get_logger().info(message)


def log_error(message: str) -> None:
    """Log an error message.

    Args:
        message: Message to log.
    """
    get_logger().error(message)


def log_warning(message: str) -> None:
    """Log a warning message.

    Args:
        message: Message to log.
    """
    get_logger().warning(message)


def log_debug(message: str) -> None:
    """Log a debug message.

    Args:
        message: Message to log.
    """
    get_logger().debug(message)


def print_error(message: str) -> None:
    """Print an error message to console.

    Args:
        message: Message to print.
    """
    stderr_console.print(f"[bold red]Error:[/bold red] {message}")


def print_success(message: str) -> None:
    """Print a success message to console.

    Args:
        message: Message to print.
    """
    console.print(f"[bold green]✓[/bold green] {message}")


def print_warning(message: str) -> None:
    """Print a warning message to console.

    Args:
        message: Message to print.
    """
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message to console.

    Args:
        message: Message to print.
    """
    console.print(f"[bold blue]i[/bold blue] {message}")
