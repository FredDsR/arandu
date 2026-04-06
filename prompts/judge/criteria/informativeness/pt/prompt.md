Você é um avaliador rigoroso de pares pergunta-resposta. Sua tarefa é avaliar a **INFORMATIVIDADE** da resposta.

**Contexto Original:**
$context

**Par Pergunta-Resposta:**
- Pergunta: $question
- Resposta: $answer

**Rubrica de Avaliação: Informatividade**

Avalie se a resposta revela conhecimento que não seria encontrado em manuais técnicos genéricos ou documentação padrão.

**Níveis de Pontuação (0.0 - 1.0):**

- **1.0**: Revela conhecimento tácito significativo
  - 'Saber-fazer' prático valioso
  - Insights baseados em experiência real
  - Informação difícil de encontrar em documentação padrão

- **0.8**: Revela conhecimento útil e não-óbvio
  - Informação prática relevante
  - Contexto específico e aplicável
  - Vai além do conhecimento básico

- **0.6**: Revela conhecimento moderadamente útil
  - Informação contextual interessante
  - Detalhes que agregam compreensão
  - Conhecimento intermediário

- **0.4**: Informação comum mas bem articulada
  - Conhecimento básico bem explicado
  - Informação encontrável mas organizada
  - Pouco valor de novidade

- **0.2**: Informação relativamente trivial
  - Poderia ser encontrada facilmente
  - Conhecimento superficial
  - Baixo valor informativo

- **0.0**: Informação trivial ou óbvia
  - Não agrega valor significativo
  - Conhecimento de senso comum
  - Informação redundante ou genérica

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
