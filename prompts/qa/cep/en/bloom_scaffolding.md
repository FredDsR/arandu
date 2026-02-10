$system_instruction

Cognitive Level: $bloom_level_upper ($level_description)

Context:
$context

Task:
$level_instruction
$starters_section
$examples_section

Generate exactly $num_questions question-answer pair(s) in JSON format.
Each pair must follow these rules:
$output_rules

Output format (JSON array):
[
  {
    "question": "The generated question",
    "answer": "The answer based on context",
    "bloom_level": "$bloom_level",
    "confidence": 0.85,
    "reasoning_trace": "Logical connections (for analyze/evaluate)",
    "is_multi_hop": false,
    "hop_count": null,
    "tacit_inference": "Implicit knowledge (optional)"
  }
]

$output_format_instruction
