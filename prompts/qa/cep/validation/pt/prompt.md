$system_instruction

Contexto Original:
$context

Par Pergunta-Resposta a Avaliar:
- Pergunta: $question
- Resposta: $answer
- Nível Bloom Declarado: $bloom_level ($bloom_level_desc)

$validation_instruction

Critérios de Avaliação:

1. FAITHFULNESS (Fidelidade): $faithfulness_desc
   Rubrica:
$rubric_faithfulness

2. BLOOM_CALIBRATION (Calibração de Bloom): $bloom_desc
   O nível declarado é "$bloom_level": $bloom_level_desc
   Rubrica:
$rubric_bloom_calibration

3. INFORMATIVENESS (Informatividade): $informativeness_desc
   Rubrica:
$rubric_informativeness

$output_format_instruction
