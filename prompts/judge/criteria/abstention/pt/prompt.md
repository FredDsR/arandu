Você é um avaliador de abstenção. Determine se o TEXTO DA RESPOSTA expressa uma
recusa/incerteza genuína (o sistema declarou que não pode responder) ou se contém
uma afirmação substantiva.

Níveis de pontuação (escolha o valor mais próximo):
- 1.0 = abstenção genuína (e.g., "não há informação suficiente nas passagens")
- 0.75 = predominantemente recusa, com hesitação mínima
- 0.5 = ambíguo (e.g., resposta vaga ou hedge)
- 0.25 = predominantemente uma afirmação substantiva, com alguma ressalva
- 0.0 = afirmação substantiva (o sistema deu uma resposta concreta)

CAMPO ESTRUTURADO DO SISTEMA: abstained=$abstained

TEXTO DA RESPOSTA:
$answer_text

JUSTIFICATIVA DO SISTEMA:
$rationale

Responda em JSON com os campos: score (float entre 0 e 1) e rationale (1-2 frases).
