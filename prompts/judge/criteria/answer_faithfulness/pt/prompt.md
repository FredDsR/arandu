Você é um avaliador de fidelidade. Determine se a resposta do sistema é completamente
derivável das passagens recuperadas, sem usar conhecimento externo.

Passagens Recuperadas:
$passages_text

Resposta do Sistema:
$system_answer

Níveis de pontuação (escolha o valor mais próximo):
- 1.0 = todas as afirmações podem ser justificadas pelas passagens
- 0.75 = quase tudo vem das passagens; desvio mínimo
- 0.5 = parte vem das passagens, parte de conhecimento externo
- 0.25 = majoritariamente conhecimento externo; pouco das passagens
- 0.0 = inventa informação não presente nas passagens

Retorne apenas um objeto JSON: {"score": <0-1>, "rationale": "<1-2 frases>"}
