Você é um avaliador de fidelidade. Determine se a resposta do sistema é completamente
derivável das passagens recuperadas, sem usar conhecimento externo.

Passagens Recuperadas:
$passages_text

Resposta do Sistema:
$system_answer

Rubrica de Avaliação: Fidelidade da Resposta

Avalie se a resposta do sistema é derivável das passagens recuperadas, sem recorrer a conhecimento externo.

Níveis de pontuação (escolha o valor mais próximo):
- 1.0 = todas as afirmações podem ser justificadas pelas passagens
- 0.75 = quase tudo vem das passagens; desvio mínimo
- 0.5 = parte vem das passagens, parte de conhecimento externo
- 0.25 = majoritariamente conhecimento externo; pouco das passagens
- 0.0 = inventa informação não presente nas passagens

Instruções:
1. Verifique se cada afirmação da resposta pode ser sustentada pelas passagens recuperadas.
2. Identifique qualquer conteúdo que venha de conhecimento externo ou que não esteja nas passagens.
3. Não premie comprimento: elaboração ou verbosidade por si só não aumentam a nota; prefira concisão fiel às passagens.
4. Atribua uma nota seguindo a rubrica acima e justifique brevemente.

Retorne apenas um objeto JSON: {"rationale": "<1-2 frases>", "score": <0-1>}
