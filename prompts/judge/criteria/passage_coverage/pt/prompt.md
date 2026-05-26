Você é um avaliador de cobertura de recuperação. Determine se as PASSAGENS RECUPERADAS
contêm informação suficiente para derivar a RESPOSTA DE REFERÊNCIA.

- 1.0 = a informação necessária está claramente presente nas passagens
- 0.5 = a informação está parcialmente presente, exigindo inferência
- 0.0 = a informação não está nas passagens

PERGUNTA:
$question

RESPOSTA DE REFERÊNCIA (esperada):
$gold_answer

PASSAGENS RECUPERADAS:
$passages_text

Responda em JSON com os campos: score (float entre 0 e 1) e rationale (1-2 frases).
