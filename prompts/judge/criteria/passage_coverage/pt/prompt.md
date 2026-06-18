Você é um avaliador de cobertura de recuperação. Determine se as passagens recuperadas
contêm informação suficiente para derivar a resposta de referência.

Pergunta:
$question

Resposta de Referência (esperada):
$gold_answer

Passagens Recuperadas:
$passages_text

Rubrica de Avaliação: Cobertura de Recuperação

Avalie se as passagens recuperadas contêm informação suficiente para derivar a resposta de referência.

Níveis de pontuação (escolha o valor mais próximo):
- 1.0 = a informação necessária está clara e completamente presente nas passagens
- 0.75 = presente, mas exige juntar ou inferir levemente a partir das passagens
- 0.5 = parcialmente presente; falta parte da informação necessária
- 0.25 = apenas tangencialmente relacionada; quase nada do necessário está presente
- 0.0 = a informação não está nas passagens

Instruções:
1. Leia a resposta de referência e identifique a informação necessária para derivá-la.
2. Verifique se essa informação está presente nas passagens recuperadas, ainda que exija juntar trechos.
3. Avalie a cobertura das passagens em relação à referência, não a qualidade de redação.
4. Atribua uma nota seguindo a rubrica acima e justifique brevemente.

Retorne apenas um objeto JSON: {"rationale": "<1-2 frases>", "score": <0-1>}
