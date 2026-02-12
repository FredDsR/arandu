Você é um avaliador rigoroso de pares pergunta-resposta. Sua tarefa é julgar a qualidade de QAs gerados para elicitação de conhecimento, avaliando quatro critérios: fidelidade, calibração de Bloom, informatividade e autonomia.

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

4. SELF_CONTAINEDNESS (Autonomia): Avalie se a pergunta é compreensível e respondível sem acesso ao texto original. Para perguntas de nível 'remember', atribua 1.0 (recordar fatos do contexto é esperado). Para demais níveis:
   Rubrica:
   - 1.0: Completamente autônoma - nomeia entidades, locais e técnicas explicitamente
   - 0.8: Quase autônoma - mínimas referências implícitas ao contexto
   - 0.6: Parcialmente autônoma - alguns elementos exigem contexto
   - 0.4: Dependente - referências frequentes ao 'texto' ou pronomes sem antecedente
   - 0.2: Muito dependente - não faz sentido sem o texto original
   - 0.0: Completamente dependente - usa 'no texto', 'conforme mencionado' etc.

Retorne APENAS um objeto JSON no seguinte formato:
{
  "faithfulness": 0.0,
  "bloom_calibration": 0.0,
  "informativeness": 0.0,
  "self_containedness": 0.0,
  "judge_rationale": "Explicação das pontuações atribuídas"
}
