**Evaluation Rubric: Self-Containedness**

Evaluate whether the question is understandable and answerable without access to the original text.

**IMPORTANT:** For 'remember' level questions, automatically assign 1.0 (recalling facts from context is expected).

**Scoring Levels (0.0 - 1.0):**

- **1.0**: Completely self-contained
  - Explicitly names entities, places, and techniques
  - Provides all necessary context in the question itself
  - Understandable without the original text

- **0.8**: Nearly self-contained
  - Minimal implicit references to context
  - Sufficient information for general understanding
  - Small details may require context

- **0.6**: Partially self-contained
  - Some elements require context
  - Indirect references present
  - Partially understandable without context

- **0.4**: Dependent
  - Frequent references to 'the text'
  - Pronouns without clear antecedent
  - Difficult to understand without original text

- **0.2**: Very dependent
  - Does not make sense without original text
  - Multiple contextual references
  - Insufficient information in the question

- **0.0**: Completely dependent
  - Uses 'in the text', 'as mentioned', etc.
  - Impossible to answer without access to context
  - Direct references to the original text
