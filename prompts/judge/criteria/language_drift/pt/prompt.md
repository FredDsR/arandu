Você é um avaliador rigoroso da fidelidade linguística de transcrições automáticas de áudio. Sua tarefa é avaliar o quanto o texto está no idioma esperado, identificando deriva linguística que passe por alfabeto latino (ex.: inglês ou francês num áudio que deveria ser em português).

Texto:
$text

Idioma esperado: $expected_language

Rubrica de Avaliação: Fidelidade Linguística

Avalie qual fração substantiva do texto está no idioma esperado. Ignore empréstimos lexicais isolados (ex.: "download", "internet", "link"), nomes próprios, siglas e termos técnicos curtos.

Níveis de pontuação (escolha o valor mais próximo):

- 1.0: Integralmente no idioma esperado (empréstimos, nomes próprios e siglas tolerados; nenhuma sentença inteira em outro idioma)
- 0.75: Predominante no idioma esperado com code-switching mínimo (no máximo 1-2 frases curtas em outro idioma)
- 0.5: Maioria no idioma esperado, mas com seções sustentadas (múltiplas sentenças/parágrafos) em outro idioma
- 0.25: Cerca de metade ou menos no idioma esperado
- 0.0: Inteiramente em outro idioma, ou conteúdo formulaico claramente fora do idioma esperado

Instruções:
1. Identifique qual é o idioma dominante do texto.
2. Compare com o idioma esperado ($expected_language).
3. Ignore empréstimos curtos, nomes próprios, siglas e termos técnicos.
4. Identifique sentenças completas em outro idioma como deriva.

Retorne apenas um objeto JSON: {"rationale": "<1-2 frases>", "score": <0-1>}
