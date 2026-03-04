# Composable Judge Pipeline

## Overview

The composable judge pipeline implements a G-Eval-style LLM-as-a-Judge validation system where each criterion is evaluated independently in separate LLM calls to avoid reasoning overlap.

## Architecture

```
JudgeCriterion (Protocol)
├── FileCriterion (Implementation)
│   ├── Loads rubric and prompt from files
│   ├── evaluate() -> CriterionScore
│
JudgeRegistry
├── Manages available criteria
├── get_criterion(name) -> JudgeCriterion
├── get_criteria(set_name) -> list[JudgeCriterion]
│
JudgePipeline
├── Orchestrates multi-criterion evaluation
├── Configurable weights per criterion
├── evaluate() -> ValidationScore
```

## Configuration

### Environment Variables

```bash
# Enable/disable composable pipeline (default: true)
ARANDU_JUDGE_USE_COMPOSABLE_PIPELINE=true

# Language for criterion prompts (default: pt)
ARANDU_JUDGE_LANGUAGE=pt

# LLM settings
ARANDU_JUDGE_TEMPERATURE=0.3
ARANDU_JUDGE_MAX_TOKENS=2048
```

### Programmatic Configuration

```python
from arandu.config import JudgeConfig

judge_config = JudgeConfig(
    use_composable_pipeline=True,  # Use new pipeline
    language="pt",                  # Criterion prompts language
    temperature=0.3,                # Low for consistent evaluation
    max_tokens=2048,                # Increase for thinking models
)
```

## Usage

### With QAValidator (Automatic)

The QAValidator automatically uses the composable pipeline when enabled:

```python
from arandu.config import CEPConfig, JudgeConfig
from arandu.core.cep.validator import QAValidator
from arandu.core.llm_client import LLMClient, LLMProvider

# Create LLM client for validation
validator_client = LLMClient(
    provider=LLMProvider.OLLAMA,
    model_id="qwen3:14b",
)

# Load configurations
cep_config = CEPConfig()
judge_config = JudgeConfig(use_composable_pipeline=True)

# Create validator
validator = QAValidator(
    validator_client=validator_client,
    cep_config=cep_config,
    judge_config=judge_config,
)

# Validate QA pair
validated_pair = validator.validate(qa_pair, context)
```

### Direct Pipeline Usage

You can also use the judge pipeline directly:

```python
from arandu.core.judge import JudgeRegistry, JudgePipeline
from arandu.core.llm_client import LLMClient, LLMProvider

# Create LLM client
llm_client = LLMClient(
    provider=LLMProvider.OLLAMA,
    model_id="qwen3:14b",
)

# Create registry
registry = JudgeRegistry(
    llm_client=llm_client,
    language="pt",
    temperature=0.3,
    max_tokens=2048,
)

# Get criteria for CEP validation
criteria = registry.get_criteria("cep_validation")

# Define weights
weights = {
    "faithfulness": 0.30,
    "bloom_calibration": 0.25,
    "informativeness": 0.25,
    "self_containedness": 0.20,
}

# Create pipeline
pipeline = JudgePipeline(criteria=criteria, weights=weights)

# Evaluate
validation_score = pipeline.evaluate(
    context="Source text...",
    question="Question text?",
    answer="Answer text.",
    bloom_calibration={"bloom_level": "analyze", "bloom_level_desc": "..."},
    self_containedness={"bloom_level": "analyze"},
)
```

## Adding New Criteria

### 1. Create Criterion Files

Create a directory structure:
```
prompts/judge/criteria/
└── your_criterion/
    ├── pt/
    │   ├── rubric.md
    │   └── prompt.md
    └── en/
        ├── rubric.md
        └── prompt.md
```

### 2. Rubric File (`rubric.md`)

Define the scoring rubric with levels 0.0-1.0:

```markdown
**Rubrica de Avaliação: Your Criterion**

Descrição do critério...

**Níveis de Pontuação (0.0 - 1.0):**

- **1.0**: Description...
- **0.8**: Description...
- **0.6**: Description...
- **0.4**: Description...
- **0.2**: Description...
- **0.0**: Description...
```

### 3. Prompt File (`prompt.md`)

Define the evaluation prompt template:

```markdown
Você é um avaliador rigoroso. Sua tarefa é avaliar **YOUR CRITERION**.

**Contexto:**
$context

**Pergunta-Resposta:**
- Pergunta: $question
- Resposta: $answer

**Critério:**
$rubric

**Instruções:**
1. Leia atentamente...
2. Avalie segundo a rubrica...
3. Atribua pontuação de 0.0 a 1.0

**Retorne APENAS JSON:**
```json
{
  "score": 0.0,
  "rationale": "Explicação..."
}
```
```

### 4. Register Criterion Set

Add to `JudgeRegistry.CRITERION_SETS`:

```python
CRITERION_SETS = {
    "cep_validation": [...],
    "your_pipeline": ["your_criterion", "another_criterion"],
}
```

## Benefits

### 1. Avoids Reasoning Overlap

Each criterion is evaluated independently:
- **Problem**: Single-call evaluation can mix reasoning (e.g., faithfulness reasoning influencing informativeness score)
- **Solution**: Separate LLM calls ensure each criterion is judged in isolation

### 2. Composable Across Pipelines

The same framework can be reused:
- **CEP validation**: faithfulness, bloom_calibration, informativeness, self_containedness
- **KG validation**: factual_accuracy, completeness, consistency (future)
- **Custom pipelines**: Define your own criterion sets

### 3. Better Transparency

Per-criterion results:
```python
validation_score.criterion_scores["faithfulness"]
# CriterionScore(
#     criterion_name="faithfulness",
#     score=0.9,
#     rationale="Answer is well-grounded...",
#     thinking="<internal reasoning>",
# )
```

### 4. Configurable Criteria

Criteria are defined as data (prompt files), not code:
- **Easy to modify**: Edit rubric without changing code
- **Language support**: Simple to add new languages
- **Version control**: Track prompt changes like code

## Legacy Mode

For backward compatibility, disable the composable pipeline:

```python
judge_config = JudgeConfig(use_composable_pipeline=False)
```

This uses the original single-call validation approach.

## Performance Considerations

### Composable Mode
- **LLM Calls**: 4 calls per validation (one per criterion)
- **Token Usage**: ~4x single-call (but more accurate)
- **Parallelization**: Can evaluate criteria in parallel (future optimization)

### Legacy Mode
- **LLM Calls**: 1 call per validation
- **Token Usage**: Lower
- **Accuracy**: May have reasoning overlap

## Best Practices

1. **Use composable mode for production**: More accurate despite higher cost
2. **Adjust weights**: Fine-tune based on your use case
3. **Monitor thinking traces**: Debug unexpected scores using per-criterion thinking
4. **Cache results**: Store ValidationScore with criterion_scores for analysis
5. **Version prompts**: Track changes to rubrics and prompts in git

## References

- [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634)
- [deepeval G-Eval metric](https://docs.confident-ai.com/docs/metrics-llm-evals)
