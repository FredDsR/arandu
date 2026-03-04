"""Shared console instance for Arandu.

Provides a single console instance to be used across the application
for consistent output behavior.
"""

from __future__ import annotations

from rich.console import Console

# Global console instances
console = Console()
stderr_console = Console(stderr=True)
