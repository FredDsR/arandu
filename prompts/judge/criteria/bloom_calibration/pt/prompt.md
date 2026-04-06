Você é um avaliador rigoroso de pares pergunta-resposta. Sua tarefa é avaliar a **CALIBRAÇÃO DE BLOOM** da pergunta.

**Contexto Original:**
$context

**Par Pergunta-Resposta:**
- Pergunta: $question
- Resposta: $answer
- Nível Bloom Declarado: $bloom_level ($bloom_level_desc)

**Rubrica de Avaliação: Calibração de Bloom**

Avalie se a pergunta realmente exige o nível cognitivo declarado segundo a Taxonomia de Bloom.

**Níveis de Pontuação (0.0 - 1.0):**

- **1.0**: Pergunta perfeitamente calibrada
  - Exige exatamente o nível cognitivo declarado
  - Não pode ser respondida adequadamente com nível inferior

- **0.8**: Pergunta bem calibrada
  - Exige predominantemente o nível declarado
  - Mínima sobreposição com níveis adjacentes

- **0.6**: Pergunta razoavelmente calibrada
  - Exige o nível declarado com alguma sobreposição
  - Aspectos do nível correto estão presentes

- **0.4**: Pergunta subcalibrada
  - Exige nível cognitivo menor que o declarado
  - Pode ser respondida com processos cognitivos mais simples

- **0.2**: Pergunta significativamente descalibrada
  - Nível cognitivo exigido muito diferente do declarado

- **0.0**: Pergunta totalmente descalibrada
  - Nível cognitivo completamente diferente do declarado
  - Não há correspondência com o nível proposto

**Instruções:**
1. Leia atentamente a pergunta e identifique o nível cognitivo que ela realmente exige
2. Compare o nível exigido com o nível declarado ($bloom_level)
3. Considere se a resposta requer apenas recordação, compreensão, aplicação, análise, avaliação ou criação
4. Atribua uma pontuação de 0.0 a 1.0 seguindo a rubrica acima
5. Forneça uma justificativa explicando a correspondência ou descalibração

**Retorne APENAS um objeto JSON no seguinte formato:**
```json
{
  "score": 0.0,
  "rationale": "Explicação da pontuação atribuída"
}
```
