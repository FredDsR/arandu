"""Tests for console utilities."""

from __future__ import annotations

from gtranscriber.utils.console import console, stderr_console


class TestConsoleInstances:
    """Tests for console instances."""

    def test_console_instance_exists(self) -> None:
        """Test that console instance is created."""
        assert console is not None
        assert hasattr(console, "print")

    def test_stderr_console_instance_exists(self) -> None:
        """Test that stderr_console instance is created."""
        assert stderr_console is not None
        assert hasattr(stderr_console, "print")

    def test_consoles_are_different_instances(self) -> None:
        """Test that console and stderr_console are different instances."""
        assert console is not stderr_console
