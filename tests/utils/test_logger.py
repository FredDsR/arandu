"""Tests for logger utilities."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from arandu.utils.logger import (
    get_logger,
    log_debug,
    log_error,
    log_info,
    log_warning,
    print_error,
    print_info,
    print_success,
    print_warning,
    setup_logging,
)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_default(self) -> None:
        """Test setup_logging with default parameters."""
        logger = setup_logging()

        assert logger is not None
        assert logger.name == "arandu"
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0

    def test_setup_logging_custom_level(self) -> None:
        """Test setup_logging with custom log level."""
        logger = setup_logging(level=logging.DEBUG)

        assert logger.level == logging.DEBUG

    def test_setup_logging_clears_handlers(self) -> None:
        """Test that setup_logging clears existing handlers."""
        # Setup first time
        logger1 = setup_logging()
        handlers_count_1 = len(logger1.handlers)

        # Setup again
        logger2 = setup_logging()
        handlers_count_2 = len(logger2.handlers)

        # Should have same number of handlers (old ones cleared)
        assert handlers_count_1 == handlers_count_2
        assert logger1 is logger2  # Same logger instance

    def test_setup_logging_show_time_false(self) -> None:
        """Test setup_logging with show_time=False."""
        logger = setup_logging(show_time=False)

        assert logger is not None
        assert len(logger.handlers) > 0

    def test_setup_logging_show_path_true(self) -> None:
        """Test setup_logging with show_path=True."""
        logger = setup_logging(show_path=True)

        assert logger is not None
        assert len(logger.handlers) > 0


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_no_name(self) -> None:
        """Test get_logger without name parameter."""
        logger = get_logger()

        assert logger is not None
        assert logger.name == "arandu"

    def test_get_logger_with_name(self) -> None:
        """Test get_logger with custom name."""
        logger = get_logger("custom")

        assert logger is not None
        assert logger.name == "arandu.custom"

    def test_get_logger_different_names(self) -> None:
        """Test that different names return different loggers."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        assert logger1.name != logger2.name
        assert "module1" in logger1.name
        assert "module2" in logger2.name


class TestLogFunctions:
    """Tests for convenience logging functions."""

    @patch("arandu.utils.logger.get_logger")
    def test_log_info(self, mock_get_logger: MagicMock) -> None:
        """Test log_info function."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        log_info("Test info message")

        mock_get_logger.assert_called_once()
        mock_logger.info.assert_called_once_with("Test info message")

    @patch("arandu.utils.logger.get_logger")
    def test_log_error(self, mock_get_logger: MagicMock) -> None:
        """Test log_error function."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        log_error("Test error message")

        mock_get_logger.assert_called_once()
        mock_logger.error.assert_called_once_with("Test error message")

    @patch("arandu.utils.logger.get_logger")
    def test_log_warning(self, mock_get_logger: MagicMock) -> None:
        """Test log_warning function."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        log_warning("Test warning message")

        mock_get_logger.assert_called_once()
        mock_logger.warning.assert_called_once_with("Test warning message")

    @patch("arandu.utils.logger.get_logger")
    def test_log_debug(self, mock_get_logger: MagicMock) -> None:
        """Test log_debug function."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        log_debug("Test debug message")

        mock_get_logger.assert_called_once()
        mock_logger.debug.assert_called_once_with("Test debug message")


class TestPrintFunctions:
    """Tests for console print functions."""

    @patch("arandu.utils.logger.stderr_console")
    def test_print_error(self, mock_stderr_console: MagicMock) -> None:
        """Test print_error function."""
        print_error("Test error")

        mock_stderr_console.print.assert_called_once()
        call_args = mock_stderr_console.print.call_args[0][0]
        assert "Error:" in call_args
        assert "Test error" in call_args

    @patch("arandu.utils.logger.console")
    def test_print_success(self, mock_console: MagicMock) -> None:
        """Test print_success function."""
        print_success("Test success")

        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert "✓" in call_args
        assert "Test success" in call_args

    @patch("arandu.utils.logger.console")
    def test_print_warning(self, mock_console: MagicMock) -> None:
        """Test print_warning function."""
        print_warning("Test warning")

        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert "⚠" in call_args
        assert "Test warning" in call_args

    @patch("arandu.utils.logger.console")
    def test_print_info(self, mock_console: MagicMock) -> None:
        """Test print_info function."""
        print_info("Test info")

        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert "i" in call_args
        assert "Test info" in call_args
