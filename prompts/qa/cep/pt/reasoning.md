Analise o seguinte par pergunta-resposta e forneça informações de raciocínio.

Contexto:
$context

Pergunta: $question
Resposta: $answer
Nível Bloom: $bloom_level

Tarefas:
1. $reasoning_instruction
2. Determine se a resposta requer conectar informações de partes distantes do texto (multi-hop).
3. Se for multi-hop, indique quantos "saltos" de raciocínio são necessários (1-5).
4. $tacit_instruction

Retorne APENAS um objeto JSON no seguinte formato:
{
  "reasoning_trace": "Fato A + Fato B → Conclusão",
  "is_multi_hop": true/false,
  "hop_count": 2,
  "tacit_inference": "Conhecimento implícito identificado"
}
