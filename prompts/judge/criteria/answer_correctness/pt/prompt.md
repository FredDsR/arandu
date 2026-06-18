Você é um avaliador de qualidade de respostas. Compare a resposta do sistema com a
resposta de referência e atribua uma nota entre 0 e 1.

Pergunta:
$question

Resposta de Referência:
$gold_answer

Resposta do Sistema:
$system_answer

Níveis de pontuação (escolha o valor mais próximo):
- 1.0 = transmite exatamente a mesma informação da referência
- 0.75 = correta, com omissões ou diferenças menores
- 0.5 = parcialmente correta, ou correta mas com omissões importantes
- 0.25 = majoritariamente incorreta, com algum acerto
- 0.0 = incorreta ou contradiz a referência

Retorne apenas um objeto JSON: {"score": <0-1>, "rationale": "<1-2 frases>"}
