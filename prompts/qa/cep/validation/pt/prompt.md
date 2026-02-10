Você é um avaliador rigoroso de pares pergunta-resposta. Sua tarefa é julgar a qualidade de QAs gerados para elicitação de conhecimento, avaliando três critérios: fidelidade, calibração de Bloom e informatividade.

Contexto Original:
$context

Par Pergunta-Resposta a Avaliar:
- Pergunta: $question
- Resposta: $answer
- Nível Bloom Declarado: $bloom_level ($bloom_level_desc)

Analise o par pergunta-resposta fornecido e avalie cada critério. Forneça uma pontuação de 0.0 a 1.0 para cada critério e uma justificativa breve.

Critérios de Avaliação:

1. FAITHFULNESS (Fidelidade): Avalie se a resposta é fundamentada no contexto (1.0) ou se contém alucinações (0.0).
   Rubrica:
   - 1.0: Resposta completamente fundamentada no texto - todas as informações podem ser verificadas no contexto
   - 0.8: Resposta bem fundamentada com inferências mínimas e razoáveis
   - 0.6: Resposta principalmente fundamentada, mas com algumas inferências não-triviais
   - 0.4: Resposta parcialmente fundamentada com inferências significativas ou informações não verificáveis
   - 0.2: Resposta fracamente fundamentada - maior parte é inferência ou não verificável
   - 0.0: Resposta não fundamentada, alucinada ou contraditória ao contexto

2. BLOOM_CALIBRATION (Calibração de Bloom): Avalie se a pergunta realmente exige o nível cognitivo declarado.
   O nível declarado é "$bloom_level": $bloom_level_desc
   Rubrica:
   - 1.0: Pergunta perfeitamente calibrada - exige exatamente o nível cognitivo declarado
   - 0.8: Pergunta bem calibrada - exige predominantemente o nível declarado
   - 0.6: Pergunta razoavelmente calibrada - exige o nível declarado com alguma sobreposição
   - 0.4: Pergunta subcalibrada - exige nível cognitivo menor que o declarado
   - 0.2: Pergunta significativamente descalibrada
   - 0.0: Pergunta totalmente descalibrada - nível cognitivo completamente diferente

3. INFORMATIVENESS (Informatividade): Avalie se a resposta revela conhecimento que não seria encontrado em manuais técnicos genéricos.
   Rubrica:
   - 1.0: Revela conhecimento tácito significativo - 'saber-fazer' prático ou insight valioso
   - 0.8: Revela conhecimento útil e não-óbvio - informação prática relevante
   - 0.6: Revela conhecimento moderadamente útil - informação contextual interessante
   - 0.4: Informação comum mas bem articulada - conhecimento básico bem explicado
   - 0.2: Informação relativamente trivial - poderia ser encontrada facilmente
   - 0.0: Informação trivial ou óbvia - não agrega valor significativo

Retorne APENAS o objeto JSON com as pontuações e justificativa, sem texto adicional.
