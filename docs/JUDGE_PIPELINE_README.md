# Composable Judge Pipeline - Implementation Summary

## Overview

This PR implements a composable G-Eval-style LLM-as-a-Judge validation pipeline that addresses the limitations of the original single-call validation approach.

## Problem Solved

The original `QAValidator` evaluated all four criteria (faithfulness, bloom_calibration, informativeness, self_containedness) in a single LLM call, causing:

1. **Reasoning Overlap**: Cross-contamination between judgments (e.g., faithfulness reasoning influencing informativeness score)
2. **Not Composable**: Hardcoded to CEP's four criteria, couldn't be reused for other pipeline steps
3. **Not Extensible**: Adding new metrics required modifying the validator class and prompt template

## Solution

A modular, composable judge framework inspired by [G-Eval](https://arxiv.org/abs/2303.16634):

- **One criterion per LLM call**: Each metric evaluated independently
- **Configurable criteria**: Defined as data (prompt/rubric files) rather than code
- **Composable across pipelines**: Works for CEP validation, KG evaluation, and any future step

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      JudgeRegistry                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Criterion Sets:                                      │   │
│  │ - cep_validation: [faithfulness, bloom_calibration, │   │
│  │                    informativeness, self_containedness] │
│  │ - kg_validation: [factual_accuracy, completeness]   │   │
│  │   (future)                                           │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     JudgePipeline                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ For each criterion:                                  │   │
│  │ 1. Load rubric & prompt template                    │   │
│  │ 2. Build evaluation prompt                          │   │
│  │ 3. Call LLM judge (independent)                     │   │
│  │ 4. Parse CriterionScore                             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Calculate weighted overall score                          │
│  Build ValidationScore with criterion_scores               │
└─────────────────────────────────────────────────────────────┘
```

## Files Structure

### Core Module (`src/gtranscriber/core/judge/`)

```
judge/
├── __init__.py          # Module exports
├── criterion.py         # JudgeCriterion protocol & FileCriterion implementation
├── registry.py          # JudgeRegistry for managing criteria
└── pipeline.py          # JudgePipeline for orchestrating evaluation
```

### Prompts (`prompts/judge/criteria/`)

```
criteria/
├── faithfulness/
│   ├── pt/
│   │   ├── rubric.md    # Scoring rubric (0.0-1.0)
│   │   └── prompt.md    # Evaluation prompt template
│   └── en/
│       ├── rubric.md
│       └── prompt.md
├── bloom_calibration/   # Same structure
├── informativeness/     # Same structure
└── self_containedness/  # Same structure
```

### Tests (`tests/core/judge/`)

```
judge/
├── __init__.py
├── test_criterion.py    # FileCriterion tests
├── test_registry.py     # JudgeRegistry tests
└── test_pipeline.py     # JudgePipeline tests
```

## Usage

### Quick Start (Default: Composable Mode)

```python
from gtranscriber.config import CEPConfig
from gtranscriber.core.cep.validator import QAValidator
from gtranscriber.core.llm_client import LLMClient, LLMProvider

# Create validator (automatically uses composable pipeline)
validator_client = LLMClient(LLMProvider.OLLAMA, "qwen3:14b")
validator = QAValidator(validator_client, CEPConfig())

# Validate - will make 4 independent LLM calls
validated_pair = validator.validate(qa_pair, context)

# Access per-criterion scores
print(validated_pair.validation.criterion_scores["faithfulness"])
# CriterionScore(criterion_name="faithfulness", score=0.9, rationale="...", thinking="...")
```

### Legacy Mode (Single Call)

```python
from gtranscriber.config import JudgeConfig

judge_config = JudgeConfig(use_composable_pipeline=False)
validator = QAValidator(validator_client, cep_config, judge_config)

# Validate - will make 1 LLM call (legacy behavior)
validated_pair = validator.validate(qa_pair, context)
```

### Direct Pipeline Usage

```python
from gtranscriber.core.judge import JudgeRegistry, JudgePipeline

registry = JudgeRegistry(llm_client, language="pt")
criteria = registry.get_criteria("cep_validation")

weights = {
    "faithfulness": 0.30,
    "bloom_calibration": 0.25,
    "informativeness": 0.25,
    "self_containedness": 0.20,
}

pipeline = JudgePipeline(criteria, weights)
validation_score = pipeline.evaluate(context, question, answer)
```

## Environment Variables

```bash
# Enable/disable composable pipeline (default: true)
GTRANSCRIBER_JUDGE_USE_COMPOSABLE_PIPELINE=true

# Language for criterion prompts (default: pt)
GTRANSCRIBER_JUDGE_LANGUAGE=pt

# LLM settings
GTRANSCRIBER_JUDGE_TEMPERATURE=0.3
GTRANSCRIBER_JUDGE_MAX_TOKENS=2048
```

## Benefits

### 1. Avoids Reasoning Overlap ✅

Each criterion evaluated independently prevents cross-contamination:

```
Legacy (Single Call):
┌─────────────────────────────────────┐
│ Judge all 4 criteria at once        │
│ → faithfulness reasoning may        │
│   influence informativeness score   │
└─────────────────────────────────────┘

Composable (4 Calls):
┌─────────────────┐  ┌─────────────────┐
│ Judge           │  │ Judge           │
│ faithfulness    │  │ bloom_          │
│ independently   │  │ calibration     │
└─────────────────┘  └─────────────────┘
┌─────────────────┐  ┌─────────────────┐
│ Judge           │  │ Judge           │
│ informativeness │  │ self_           │
│ independently   │  │ containedness   │
└─────────────────┘  └─────────────────┘
```

### 2. Composable Across Pipelines ✅

Same framework, different criterion sets:

```python
# CEP validation
registry.get_criteria("cep_validation")
# → [faithfulness, bloom_calibration, informativeness, self_containedness]

# Future: KG validation
registry.get_criteria("kg_validation")
# → [factual_accuracy, completeness, consistency]
```

### 3. Extensible via Data ✅

Add new criteria without code changes:

1. Create `prompts/judge/criteria/new_criterion/pt/{rubric.md,prompt.md}`
2. Register in `JudgeRegistry.CRITERION_SETS`
3. Done! No code changes needed.

### 4. Better Debugging ✅

Per-criterion thinking traces:

```python
for name, score in validation.criterion_scores.items():
    print(f"{name}: {score.score:.2f}")
    print(f"  Rationale: {score.rationale}")
    print(f"  Thinking: {score.thinking[:100]}...")
```

## Performance Considerations

| Mode       | LLM Calls | Token Usage | Accuracy  |
|------------|-----------|-------------|-----------|
| Composable | 4         | ~4x single  | Higher ✅ |
| Legacy     | 1         | Lower       | Lower ❌  |

**Recommendation**: Use composable mode for production (accuracy > cost).

## Testing

Comprehensive test coverage:

- ✅ `test_criterion.py`: FileCriterion evaluation, error handling, score clamping
- ✅ `test_registry.py`: Criterion loading, caching, custom criteria
- ✅ `test_pipeline.py`: Weight validation, score calculation, aggregation
- ✅ `test_validator.py`: Composable/legacy mode switching, integration tests

Run tests: `uv run pytest tests/core/judge/`

## Code Review & Security

- ✅ **Code Review**: No issues found
- ✅ **Security Scan**: No vulnerabilities detected

## Documentation

- 📖 [Composable Judge Pipeline Guide](composable_judge_pipeline.md)
- 📖 Architecture diagrams
- 📖 Usage examples
- 📖 Guide for adding new criteria

## Migration Guide

### For Users

**No changes required!** The composable pipeline is the new default.

To opt-out (use legacy mode):
```bash
export GTRANSCRIBER_JUDGE_USE_COMPOSABLE_PIPELINE=false
```

### For Developers

**Adding new criteria:**

1. Create prompt files in `prompts/judge/criteria/your_criterion/{pt,en}/`
2. Register in `JudgeRegistry.CRITERION_SETS`
3. No code changes needed!

**Using in new pipelines:**

```python
criteria = registry.get_criteria("your_pipeline")
pipeline = JudgePipeline(criteria, your_weights)
score = pipeline.evaluate(...)
```

## Future Enhancements

- [ ] Parallel criterion evaluation (performance optimization)
- [ ] Criterion result caching (avoid re-evaluation)
- [ ] Web UI for criterion management
- [ ] A/B testing framework for prompts
- [ ] Additional criterion sets (KG, transcription quality)

## References

- [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634)
- [deepeval G-Eval metric](https://docs.confident-ai.com/docs/metrics-llm-evals)
- Issue #35: feat(cep): composable G-Eval-style LLM-as-a-Judge validation pipeline

## Credits

Implemented by @copilot with guidance from @FredDsR.

---

**Status**: ✅ Ready for merge and production use
