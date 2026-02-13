Você é um avaliador rigoroso de pares pergunta-resposta. Sua tarefa é avaliar a **INFORMATIVIDADE** da resposta.

**Contexto Original:**
$context

**Par Pergunta-Resposta:**
- Pergunta: $question
- Resposta: $answer

**Critério de Avaliação:**
$rubric

**Instruções:**
1. Leia atentamente a resposta e avalie o valor informativo do conhecimento revelado
2. Considere se esta informação seria facilmente encontrada em manuais ou documentação genérica
3. Identifique se há conhecimento tácito ('saber-fazer'), insights práticos ou experiência contextual
4. Atribua uma pontuação de 0.0 a 1.0 seguindo a rubrica acima
5. Forneça uma justificativa clara sobre o valor informativo

**Retorne APENAS um objeto JSON no seguinte formato:**
```json
{
  "score": 0.0,
  "rationale": "Explicação da pontuação atribuída"
}
```
