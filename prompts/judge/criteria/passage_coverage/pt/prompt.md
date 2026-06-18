Você é um avaliador de cobertura de recuperação. Determine se as PASSAGENS RECUPERADAS
contêm informação suficiente para derivar a RESPOSTA DE REFERÊNCIA.

Níveis de pontuação (escolha o valor mais próximo):
- 1.0 = a informação necessária está clara e completamente presente nas passagens
- 0.75 = presente, mas exige juntar ou inferir levemente a partir das passagens
- 0.5 = parcialmente presente; falta parte da informação necessária
- 0.25 = apenas tangencialmente relacionada; quase nada do necessário está presente
- 0.0 = a informação não está nas passagens

PERGUNTA:
$question

RESPOSTA DE REFERÊNCIA (esperada):
$gold_answer

PASSAGENS RECUPERADAS:
$passages_text

Responda em JSON com os campos: score (float entre 0 e 1) e rationale (1-2 frases).
