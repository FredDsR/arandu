Você é um avaliador rigoroso de pares pergunta-resposta. Sua tarefa é avaliar a **AUTONOMIA** (self-containedness) da pergunta.

**Contexto Original:**
$context

**Par Pergunta-Resposta:**
- Pergunta: $question
- Resposta: $answer
- Nível Bloom: $bloom_level

**Critério de Avaliação:**
$rubric

**Instruções:**
1. **ATENÇÃO**: Se o nível Bloom for 'remember', atribua score=1.0 automaticamente (recordar fatos é esperado)
2. Para outros níveis, leia a pergunta SEM olhar para o contexto original
3. Avalie se a pergunta faz sentido e é respondível sem acesso ao texto original
4. Identifique referências ao texto ('no texto', 'conforme mencionado', pronomes sem antecedente)
5. Atribua uma pontuação de 0.0 a 1.0 seguindo a rubrica acima
6. Forneça uma justificativa clara

**Retorne APENAS um objeto JSON no seguinte formato:**
```json
{
  "score": 0.0,
  "rationale": "Explicação da pontuação atribuída"
}
```
