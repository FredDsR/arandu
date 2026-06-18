Você é um avaliador de qualidade de respostas. Compare a resposta do sistema com a
resposta de referência e atribua uma nota entre 0 e 1.

Pergunta:
$question

Resposta de Referência:
$gold_answer

Resposta do Sistema:
$system_answer

Rubrica de Avaliação: Correção da Resposta

Avalie se a resposta do sistema transmite a mesma informação que a resposta de referência, sem omissões importantes nem contradições.

Níveis de pontuação (escolha o valor mais próximo):
- 1.0 = transmite exatamente a mesma informação da referência
- 0.75 = correta, com omissões ou diferenças menores
- 0.5 = parcialmente correta, ou correta mas com omissões importantes
- 0.25 = majoritariamente incorreta, com algum acerto
- 0.0 = incorreta ou contradiz a referência

Instruções:
1. Compare a resposta do sistema com a resposta de referência, ponto a ponto.
2. Verifique se a informação essencial da referência está presente e correta, e se há contradições.
3. Não premie comprimento: elaboração ou verbosidade por si só não aumentam a nota; prefira concisão com cobertura completa.
4. Atribua uma nota seguindo a rubrica acima e justifique brevemente.

Retorne apenas um objeto JSON: {"rationale": "<1-2 frases>", "score": <0-1>}
