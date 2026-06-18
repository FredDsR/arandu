Você é um avaliador rigoroso de pares pergunta-resposta. Sua tarefa é avaliar a **AUTONOMIA** da pergunta.

O texto original NÃO é fornecido de propósito: avalie a pergunta apenas pelo que está nela própria.

**Par Pergunta-Resposta:**
- Pergunta: $question
- Resposta: $answer

**Rubrica de Avaliação: Autonomia**

Avalie se a pergunta é compreensível e respondível sem acesso ao texto original.

**Níveis de pontuação (escolha o valor mais próximo):**

- **1.0**: Completamente autônoma; nomeia entidades/locais/técnicas e fornece todo o contexto necessário na própria pergunta
- **0.75**: Quase autônoma; referências implícitas mínimas, compreensível no geral
- **0.5**: Parcialmente autônoma; alguns elementos exigem o contexto, referências indiretas presentes
- **0.25**: Dependente; referências frequentes ao texto ou pronomes sem antecedente claro
- **0.0**: Completamente dependente; usa "no texto"/"conforme mencionado", impossível responder sem o contexto

**Instruções:**
1. Avalie se a pergunta faz sentido e é respondível por si só, sem acesso ao texto original (que não foi fornecido)
2. Identifique referências ao texto ('no texto', 'conforme mencionado', pronomes sem antecedente)
3. Atribua uma pontuação de 0.0 a 1.0 seguindo a rubrica acima
4. Forneça uma justificativa clara

**Retorne APENAS um objeto JSON no seguinte formato:**
```json
{
  "score": 0.0,
  "rationale": "Explicação da pontuação atribuída"
}
```
