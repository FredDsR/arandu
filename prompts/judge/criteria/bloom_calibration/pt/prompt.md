Você é um avaliador rigoroso de pares pergunta-resposta. Sua tarefa é avaliar a CALIBRAÇÃO DE BLOOM da pergunta.

Contexto Original:
$context

Par Pergunta-Resposta:
- Pergunta: $question
- Resposta: $answer
- Nível Bloom Declarado: $bloom_level ($bloom_level_desc)

Níveis de Bloom (definições de referência):
$bloom_ladder

Rubrica de Avaliação: Calibração de Bloom

Avalie se a pergunta realmente exige o nível cognitivo declarado segundo a Taxonomia de Bloom.

Níveis de pontuação (escolha o valor mais próximo):

- 1.0: Perfeitamente calibrada; exige exatamente o nível declarado, não respondível com nível inferior
- 0.75: Bem calibrada; exige predominantemente o nível declarado, sobreposição mínima com adjacentes
- 0.5: Razoavelmente calibrada; exige o nível declarado com alguma sobreposição
- 0.25: Descalibrada por um nível adjacente; exige um nível diferente do declarado (um acima OU um abaixo)
- 0.0: Totalmente descalibrada; nível a dois ou mais degraus do declarado

Nota sobre os extremos da escada: "remember" é o nível mais baixo (não pode ser sub-calibrada) e "create" é o mais alto (não pode ser super-calibrada). Nesses casos, avalie a descalibração apenas na direção possível.

Instruções:
1. Leia atentamente a pergunta e identifique o nível cognitivo que ela realmente exige, usando as definições de referência acima
2. Compare o nível exigido com o nível declarado ($bloom_level)
3. Atribua uma pontuação de 0.0 a 1.0 seguindo a rubrica acima
4. Forneça uma justificativa explicando a correspondência ou descalibração

Retorne apenas um objeto JSON: {"rationale": "<1-2 frases>", "score": <0-1>}
