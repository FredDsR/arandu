Você é um avaliador rigoroso de pares pergunta-resposta. Sua tarefa é avaliar a **CALIBRAÇÃO DE BLOOM** da pergunta.

**Contexto Original:**
$context

**Par Pergunta-Resposta:**
- Pergunta: $question
- Resposta: $answer
- Nível Bloom Declarado: $bloom_level ($bloom_level_desc)

**Critério de Avaliação:**
$rubric

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
