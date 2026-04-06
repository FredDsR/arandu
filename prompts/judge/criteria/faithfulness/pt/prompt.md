Você é um avaliador rigoroso de pares pergunta-resposta. Sua tarefa é avaliar a **FIDELIDADE** da resposta ao contexto original.

**Contexto Original:**
$context

**Par Pergunta-Resposta:**
- Pergunta: $question
- Resposta: $answer

**Rubrica de Avaliação: Fidelidade (Faithfulness)**

Avalie se a resposta é fundamentada no contexto fornecido ou se contém alucinações/informações não verificáveis.

**Níveis de Pontuação (0.0 - 1.0):**

- **1.0**: Resposta completamente fundamentada no texto
  - Todas as informações podem ser verificadas diretamente no contexto
  - Não há inferências além do que está explicitamente declarado

- **0.8**: Resposta bem fundamentada com inferências mínimas e razoáveis
  - Informações principais verificáveis no contexto
  - Inferências lógicas diretas baseadas no texto

- **0.6**: Resposta principalmente fundamentada, mas com algumas inferências não-triviais
  - A maioria das informações verificável
  - Algumas conexões implícitas ou conhecimento de senso comum

- **0.4**: Resposta parcialmente fundamentada com inferências significativas
  - Parte da resposta verificável
  - Informações importantes não diretamente verificáveis

- **0.2**: Resposta fracamente fundamentada
  - Maior parte é inferência ou não verificável
  - Poucas conexões diretas com o contexto

- **0.0**: Resposta não fundamentada, alucinada ou contraditória
  - Informações falsas ou contraditórias ao contexto
  - Não há base factual no texto fornecido

**Instruções:**
1. Leia atentamente o contexto e o par pergunta-resposta
2. Verifique se cada afirmação na resposta pode ser encontrada ou inferida diretamente do contexto
3. Identifique qualquer alucinação, informação não verificável ou contradição
4. Atribua uma pontuação de 0.0 a 1.0 seguindo a rubrica acima
5. Forneça uma justificativa breve e clara

**Retorne APENAS um objeto JSON no seguinte formato:**
```json
{
  "score": 0.0,
  "rationale": "Explicação da pontuação atribuída"
}
```
