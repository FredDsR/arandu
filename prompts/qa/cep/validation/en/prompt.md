$system_instruction

Original Context:
$context

Question-Answer Pair to Evaluate:
- Question: $question
- Answer: $answer
- Declared Bloom Level: $bloom_level ($bloom_level_desc)

$validation_instruction

Evaluation Criteria:

1. FAITHFULNESS: $faithfulness_desc
   Rubric:
$rubric_faithfulness

2. BLOOM_CALIBRATION: $bloom_desc
   The declared level is "$bloom_level": $bloom_level_desc
   Rubric:
$rubric_bloom_calibration

3. INFORMATIVENESS: $informativeness_desc
   Rubric:
$rubric_informativeness

$output_format_instruction
