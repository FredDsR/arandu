$system_instruction

Nível Cognitivo: $bloom_level_upper ($level_description)

Contexto:
$context

Tarefa:
$level_instruction
$starters_section
$examples_section

Gere exatamente $num_questions par(es) pergunta-resposta no formato JSON.
Cada par deve seguir estas regras:
$output_rules

Formato de saída (array JSON):
[
  {
    "question": "A pergunta gerada",
    "answer": "A resposta baseada no contexto",
    "bloom_level": "$bloom_level",
    "confidence": 0.85,
    "reasoning_trace": "Conexões lógicas (para analyze/evaluate)",
    "is_multi_hop": false,
    "hop_count": null,
    "tacit_inference": "Conhecimento implícito (opcional)"
  }
]

$output_format_instruction
