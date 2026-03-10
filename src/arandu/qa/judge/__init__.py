"""Composable G-Eval-style LLM-as-a-Judge framework.

This module provides a flexible, composable framework for LLM-based evaluation
of generated content (QA pairs, knowledge graphs, etc.). Inspired by G-Eval
(https://arxiv.org/abs/2303.16634), each criterion is evaluated independently
to avoid reasoning overlap.

Key components:
- JudgeCriterion: Protocol for individual evaluation criteria
- JudgeRegistry: Manages available criteria and their configurations
- JudgePipeline: Orchestrates multi-criterion evaluation with configurable weights
"""

from __future__ import annotations

from arandu.qa.judge.criterion import JudgeCriterion
from arandu.qa.judge.pipeline import JudgePipeline
from arandu.qa.judge.registry import JudgeRegistry

__all__ = ["JudgeCriterion", "JudgePipeline", "JudgeRegistry"]
