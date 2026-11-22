Você é um avaliador rigoroso de respostas. Sua tarefa é avaliar a ABSTENÇÃO expressa no texto da resposta.

Campo Estruturado do Sistema: abstained=$abstained

Texto da Resposta:
$answer_text

Justificativa do Sistema:
$rationale

Rubrica de Avaliação: Abstenção

Avalie se o texto da resposta expressa uma recusa/incerteza genuína (o sistema declarou que não pode responder) ou uma afirmação substantiva.

Níveis de pontuação (escolha o valor mais próximo):
- 1.0 = abstenção genuína (e.g., "não há informação suficiente nas passagens")
- 0.75 = predominantemente recusa, com hesitação mínima
- 0.5 = ambíguo (e.g., resposta vaga ou hedge)
- 0.25 = predominantemente uma afirmação substantiva, com alguma ressalva
- 0.0 = afirmação substantiva (o sistema deu uma resposta concreta)

Instruções:
1. Determine se o Texto da Resposta expressa recusa ou incerteza genuína (o sistema declara que não pode responder) ou uma afirmação substantiva.
2. Quando o Texto da Resposta estiver vazio, baseie a avaliação na Justificativa do Sistema e no campo abstained: uma justificativa de recusa (o que faltava nas passagens) indica abstenção genuína.
3. Atribua a nota do nível mais próximo seguindo a rubrica acima.
4. Forneça uma justificativa breve e clara.

Retorne apenas um objeto JSON: {"rationale": "<1-2 frases>", "score": <0-1>}
