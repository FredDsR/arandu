"""Tests for text processing utilities."""

from __future__ import annotations

from arandu.utils.text import GenerateResult, extract_thinking


class TestExtractThinking:
    """Tests for extract_thinking function."""

    def test_no_tags(self) -> None:
        """Test passthrough when no think tags are present."""
        result = extract_thinking('{"key": "value"}')

        assert result.thinking is None
        assert result.content == '{"key": "value"}'

    def test_standard_thinking(self) -> None:
        """Test extraction of standard thinking block."""
        result = extract_thinking('<think>reasoning here</think>{"key": 1}')

        assert result.thinking == "reasoning here"
        assert result.content == '{"key": 1}'

    def test_empty_think_block(self) -> None:
        """Test that empty think block results in thinking=None."""
        result = extract_thinking('<think></think>{"key": 1}')

        assert result.thinking is None
        assert result.content == '{"key": 1}'

    def test_multiline_thinking(self) -> None:
        """Test thinking that spans multiple lines."""
        response = '<think>\nline 1\nline 2\nline 3\n</think>{"key": 1}'
        result = extract_thinking(response)

        assert result.thinking == "line 1\nline 2\nline 3"
        assert result.content == '{"key": 1}'

    def test_multiple_think_blocks(self) -> None:
        """Test multiple think blocks are concatenated."""
        response = "<think>first</think>middle<think>second</think>end"
        result = extract_thinking(response)

        assert result.thinking == "first\nsecond"
        assert result.content == "middle end"

    def test_empty_content_after_strip(self) -> None:
        """Test that content is empty string when only thinking is present."""
        result = extract_thinking("<think>only thinking</think>")

        assert result.thinking == "only thinking"
        assert result.content == ""

    def test_single_block_no_extra_space(self) -> None:
        """Test that single block doesn't introduce extra spaces in content."""
        result = extract_thinking('<think>reasoning</think>{"key": 1}')

        assert result.thinking == "reasoning"
        assert result.content == '{"key": 1}'

    def test_whitespace_handling(self) -> None:
        """Test whitespace between tags and content is handled."""
        result = extract_thinking('<think>  reasoning  </think>  {"key": 1}  ')

        assert result.thinking == "reasoning"
        assert result.content == '{"key": 1}'


class TestGenerateResult:
    """Tests for GenerateResult dataclass."""

    def test_defaults(self) -> None:
        """Test default thinking is None."""
        result = GenerateResult(content="hello")

        assert result.content == "hello"
        assert result.thinking is None

    def test_frozen(self) -> None:
        """Test that GenerateResult is immutable."""
        import pytest

        result = GenerateResult(content="hello", thinking="world")

        with pytest.raises(AttributeError):
            result.content = "changed"  # type: ignore[misc]
