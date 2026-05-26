You are a faithfulness evaluator. Determine whether the SYSTEM ANSWER is fully
derivable from the RETRIEVED PASSAGES, without using external knowledge.

- 1.0 = every claim in the answer can be justified by the passages
- 0.5 = part of the claims come from the passages, part from external knowledge
- 0.0 = the answer fabricates information not present in the passages

RETRIEVED PASSAGES:
$passages_text

SYSTEM ANSWER:
$system_answer

Respond in JSON with fields: score (float between 0 and 1) and rationale (1-2 sentences).
