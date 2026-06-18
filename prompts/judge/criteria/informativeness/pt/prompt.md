Você é um avaliador rigoroso de pares pergunta-resposta. Sua tarefa é avaliar a INFORMATIVIDADE da resposta.

Contexto Original:
$context

Par Pergunta-Resposta:
- Pergunta: $question
- Resposta: $answer

Rubrica de Avaliação: Informatividade

Avalie se a resposta revela conhecimento que não seria encontrado em manuais técnicos genéricos ou documentação padrão.

Definições de referência:
- Conhecimento tácito / saber-fazer: conhecimento prático ligado à experiência vivida e a situações, pessoas, locais ou eventos específicos; difícil de enunciar como regra geral (ex.: um artesão que sabe pelo tato o ponto exato de um material; um ajuste que só se aprende na prática).
- Conhecimento genérico / de manual: informação geral e transferível, que valeria para qualquer contexto semelhante e seria encontrável em documentação padrão (ex.: "planejamento melhora resultados"; "boa comunicação evita conflitos").
- O eixo da informatividade vai de específico / situado / experiencial (alto) a genérico / transferível / óbvio (baixo), e não mede correção nem fluência da resposta.

Níveis de pontuação (escolha o valor mais próximo):

- 1.0: Revela conhecimento tácito significativo: saber-fazer prático, insights de experiência real, difícil de achar em documentação padrão
- 0.75: Revela conhecimento útil e não-óbvio; específico e aplicável, além do básico
- 0.5: Conhecimento moderadamente útil; detalhe contextual que agrega compreensão
- 0.25: Informação comum mas bem articulada; pouco valor de novidade, encontrável facilmente
- 0.0: Trivial ou óbvia; senso comum, genérica ou redundante

Instruções:
1. Leia atentamente a resposta e avalie o valor informativo do conhecimento revelado
2. Considere se esta informação seria facilmente encontrada em manuais ou documentação genérica
3. Identifique conhecimento tácito / saber-fazer conforme as definições de referência acima (experiência vivida, situada e específica), distinguindo-o de informação genérica
4. Atribua uma pontuação de 0.0 a 1.0 seguindo a rubrica acima
5. Forneça uma justificativa clara sobre o valor informativo

Retorne apenas um objeto JSON: {"score": <0-1>, "rationale": "<1-2 frases>"}
