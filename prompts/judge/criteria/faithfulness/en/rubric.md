**Evaluation Rubric: Faithfulness**

Evaluate whether the answer is grounded in the provided context or contains hallucinations/unverifiable information.

**Scoring Levels (0.0 - 1.0):**

- **1.0**: Answer completely grounded in the text
  - All information can be directly verified in the context
  - No inferences beyond what is explicitly stated

- **0.8**: Well-grounded answer with minimal reasonable inferences
  - Main information verifiable in context
  - Direct logical inferences based on the text

- **0.6**: Mostly grounded answer with some non-trivial inferences
  - Most information verifiable
  - Some implicit connections or common sense knowledge

- **0.4**: Partially grounded answer with significant inferences
  - Part of the answer verifiable
  - Important information not directly verifiable

- **0.2**: Weakly grounded answer
  - Most is inference or unverifiable
  - Few direct connections to context

- **0.0**: Ungrounded, hallucinated, or contradictory answer
  - False or contradictory information
  - No factual basis in provided text
