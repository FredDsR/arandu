Você é um avaliador de cobertura de recuperação. Determine se as passagens recuperadas
contêm informação suficiente para derivar a resposta de referência.

Pergunta:
$question

Resposta de Referência (esperada):
$gold_answer

Passagens Recuperadas:
$passages_text

Níveis de pontuação (escolha o valor mais próximo):
- 1.0 = a informação necessária está clara e completamente presente nas passagens
- 0.75 = presente, mas exige juntar ou inferir levemente a partir das passagens
- 0.5 = parcialmente presente; falta parte da informação necessária
- 0.25 = apenas tangencialmente relacionada; quase nada do necessário está presente
- 0.0 = a informação não está nas passagens

Retorne apenas um objeto JSON: {"score": <0-1>, "rationale": "<1-2 frases>"}
