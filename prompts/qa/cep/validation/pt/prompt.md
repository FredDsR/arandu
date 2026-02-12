Voce e um avaliador rigoroso de pares pergunta-resposta. Sua tarefa e julgar a qualidade de QAs gerados para elicitacao de conhecimento, avaliando quatro criterios: fidelidade, calibracao de Bloom, informatividade e autonomia.

Contexto Original:
$context

Par Pergunta-Resposta a Avaliar:
- Pergunta: $question
- Resposta: $answer
- Nivel Bloom Declarado: $bloom_level ($bloom_level_desc)

Analise o par pergunta-resposta fornecido e avalie cada criterio. Forneca uma pontuacao de 0.0 a 1.0 para cada criterio e uma justificativa breve.

Criterios de Avaliacao:

1. FAITHFULNESS (Fidelidade): Avalie se a resposta e fundamentada no contexto (1.0) ou se contem alucinacoes (0.0).
   Rubrica:
   - 1.0: Resposta completamente fundamentada no texto - todas as informacoes podem ser verificadas no contexto
   - 0.8: Resposta bem fundamentada com inferencias minimas e razoaveis
   - 0.6: Resposta principalmente fundamentada, mas com algumas inferencias nao-triviais
   - 0.4: Resposta parcialmente fundamentada com inferencias significativas ou informacoes nao verificaveis
   - 0.2: Resposta fracamente fundamentada - maior parte e inferencia ou nao verificavel
   - 0.0: Resposta nao fundamentada, alucinada ou contraditoria ao contexto

2. BLOOM_CALIBRATION (Calibracao de Bloom): Avalie se a pergunta realmente exige o nivel cognitivo declarado.
   O nivel declarado e "$bloom_level": $bloom_level_desc
   Rubrica:
   - 1.0: Pergunta perfeitamente calibrada - exige exatamente o nivel cognitivo declarado
   - 0.8: Pergunta bem calibrada - exige predominantemente o nivel declarado
   - 0.6: Pergunta razoavelmente calibrada - exige o nivel declarado com alguma sobreposicao
   - 0.4: Pergunta subcalibrada - exige nivel cognitivo menor que o declarado
   - 0.2: Pergunta significativamente descalibrada
   - 0.0: Pergunta totalmente descalibrada - nivel cognitivo completamente diferente

3. INFORMATIVENESS (Informatividade): Avalie se a resposta revela conhecimento que nao seria encontrado em manuais tecnicos genericos.
   Rubrica:
   - 1.0: Revela conhecimento tacito significativo - 'saber-fazer' pratico ou insight valioso
   - 0.8: Revela conhecimento util e nao-obvio - informacao pratica relevante
   - 0.6: Revela conhecimento moderadamente util - informacao contextual interessante
   - 0.4: Informacao comum mas bem articulada - conhecimento basico bem explicado
   - 0.2: Informacao relativamente trivial - poderia ser encontrada facilmente
   - 0.0: Informacao trivial ou obvia - nao agrega valor significativo

4. SELF_CONTAINEDNESS (Autonomia): Avalie se a pergunta e compreensivel e respondivel sem acesso ao texto original. Para perguntas de nivel 'remember', atribua 1.0 (recordar fatos do contexto e esperado). Para demais niveis:
   Rubrica:
   - 1.0: Completamente autonoma - nomeia entidades, locais e tecnicas explicitamente
   - 0.8: Quase autonoma - minimas referencias implicitas ao contexto
   - 0.6: Parcialmente autonoma - alguns elementos exigem contexto
   - 0.4: Dependente - referencias frequentes ao 'texto' ou pronomes sem antecedente
   - 0.2: Muito dependente - nao faz sentido sem o texto original
   - 0.0: Completamente dependente - usa 'no texto', 'conforme mencionado' etc.

Retorne APENAS um objeto JSON no seguinte formato:
{
  "faithfulness": 0.0,
  "bloom_calibration": 0.0,
  "informativeness": 0.0,
  "self_containedness": 0.0,
  "judge_rationale": "Explicacao das pontuacoes atribuidas"
}
