Você é um avaliador rigoroso de pares pergunta-resposta. Sua tarefa é avaliar a **AUTONOMIA** (self-containedness) da pergunta.

**Contexto Original:**
$context

**Par Pergunta-Resposta:**
- Pergunta: $question
- Resposta: $answer
- Nível Bloom: $bloom_level

**Rubrica de Avaliação: Autonomia (Self-Containedness)**

Avalie se a pergunta é compreensível e respondível sem acesso ao texto original.

**IMPORTANTE:** Para perguntas de nível 'remember', atribua automaticamente 1.0 (recordar fatos do contexto é esperado).

**Níveis de Pontuação (0.0 - 1.0):**

- **1.0**: Completamente autônoma
  - Nomeia entidades, locais e técnicas explicitamente
  - Fornece todo contexto necessário na própria pergunta
  - Compreensível sem o texto original

- **0.8**: Quase autônoma
  - Mínimas referências implícitas ao contexto
  - Informação suficiente para compreensão geral
  - Pequenos detalhes podem exigir contexto

- **0.6**: Parcialmente autônoma
  - Alguns elementos exigem contexto
  - Referências indiretas presentes
  - Compreensível parcialmente sem contexto

- **0.4**: Dependente
  - Referências frequentes ao 'texto'
  - Pronomes sem antecedente claro
  - Difícil compreender sem o texto original

- **0.2**: Muito dependente
  - Não faz sentido sem o texto original
  - Múltiplas referências contextuais
  - Informação insuficiente na pergunta

- **0.0**: Completamente dependente
  - Usa 'no texto', 'conforme mencionado', etc.
  - Impossível responder sem acesso ao contexto
  - Referências diretas ao texto original

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
