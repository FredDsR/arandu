"""Script/charset match criterion for transcription quality validation.

Detects when transcription text uses unexpected character sets, such as
CJK characters when Latin script is expected for Portuguese.
"""

from __future__ import annotations

import unicodedata
from typing import Any

from arandu.shared.judge.criterion import HeuristicCriterion

# Languages that use Latin script
_LATIN_LANGS = frozenset({"pt", "en", "es", "fr", "de", "it"})


class ScriptMatchCriterion(HeuristicCriterion):
    """Evaluate whether transcription text uses the expected character set.

    Uses unicodedata.name() for robust script detection rather than
    hardcoded Unicode ranges, which miss Latin Extended-B and combining
    diacriticals used in Portuguese.

    Attributes:
        name: Criterion identifier.
        threshold: Minimum score to pass.
        max_non_latin_ratio: Maximum ratio of non-Latin characters before penalising.
    """

    name: str = "script_match"
    threshold: float = 0.6

    def __init__(
        self,
        *,
        threshold: float = 0.6,
        max_non_latin_ratio: float = 0.1,
    ) -> None:
        """Initialize script match criterion.

        Args:
            threshold: Minimum score to pass.
            max_non_latin_ratio: Maximum ratio of non-Latin characters
                for Latin-script languages.
        """
        self.threshold = threshold
        self.max_non_latin_ratio = max_non_latin_ratio

    def _check(self, **kwargs: Any) -> tuple[float, str]:
        """Check whether text uses the expected character set.

        Args:
            **kwargs: Must contain ``text`` (str). Optionally
                ``expected_language`` (str, default ``"pt"``).

        Returns:
            Tuple of (score, rationale).
        """
        text: str = kwargs["text"]
        expected_language: str = kwargs.get("expected_language", "pt")
        score, issues = self._check_script_match(text, expected_language)
        rationale = "; ".join(issues) if issues else "Text uses expected character set."
        return score, rationale

    def _check_script_match(self, text: str, expected_lang: str) -> tuple[float, list[str]]:
        """Check if text uses expected character set.

        Args:
            text: Transcription text to check.
            expected_lang: Expected language code (e.g., 'pt', 'en').

        Returns:
            Tuple of (score, issues) where score is 0.0-1.0.
        """
        if expected_lang in _LATIN_LANGS:
            latin_chars = 0
            cjk_chars = 0
            total_alpha = 0

            for c in text:
                if not c.isalpha():
                    continue
                total_alpha += 1
                name = unicodedata.name(c, "")
                if "LATIN" in name or "COMBINING" in name:
                    latin_chars += 1
                elif "CJK" in name or "HIRAGANA" in name or "KATAKANA" in name:
                    cjk_chars += 1

            if total_alpha == 0:
                return 0.5, ["no_alphabetic_content"]

            non_latin_ratio = (total_alpha - latin_chars) / total_alpha

            if cjk_chars > total_alpha * 0.5:
                return 0.0, [f"wrong_script:cjk_detected:{cjk_chars}_chars"]

            if non_latin_ratio > self.max_non_latin_ratio:
                return 0.2, [f"high_non_latin_ratio:{non_latin_ratio:.2f}"]

        return 1.0, []
