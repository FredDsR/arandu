Você é um avaliador rigoroso da fidelidade linguística de transcrições automáticas de áudio. Sua tarefa é avaliar o quanto o texto está no idioma esperado, identificando deriva linguística (language drift) que passe por alfabeto latino (ex.: inglês ou francês num áudio que deveria ser em português).

**Texto:**
$text

**Idioma esperado:** $expected_language

**Rubrica de Avaliação: Fidelidade Linguística (Language Drift)**

Avalie qual fração substantiva do texto está no idioma esperado. Ignore empréstimos lexicais isolados (ex.: "download", "internet", "link"), nomes próprios, siglas e termos técnicos curtos.

**Níveis de Pontuação (0.0 - 1.0):**

- **1.0**: Texto integralmente no idioma esperado
  - Empréstimos lexicais, nomes próprios e siglas são tolerados
  - Nenhuma sentença inteira em outro idioma

- **0.8**: Texto predominantemente no idioma esperado com code-switching mínimo
  - No máximo 1-2 frases curtas em outro idioma (ex.: citação direta, termo técnico)
  - O resto do texto é claramente no idioma esperado

- **0.6**: Mais da metade do texto no idioma esperado, mas há seções sustentadas em outro idioma
  - Múltiplas sentenças ou parágrafos em idioma estrangeiro
  - Ainda assim, o idioma dominante é o esperado

- **0.4**: Cerca de metade do texto em idioma diferente do esperado

- **0.2**: Maior parte do texto em outro idioma
  - Apenas fragmentos no idioma esperado

- **0.0**: Texto inteiramente em outro idioma ou conteúdo formulaico que claramente não é do idioma esperado
  - Exemplos: "Hello everyone, welcome to our channel", "This video is brought to you by...", "C'est parti, on va prendre notre café"

**Instruções:**
1. Identifique qual é o idioma dominante do texto.
2. Compare com o idioma esperado ($expected_language).
3. Ignore empréstimos curtos, nomes próprios, siglas e termos técnicos.
4. Flague sentenças completas em outro idioma como deriva.
5. Texto vazio ou muito curto deve receber nota 1.0 (não avaliável).

**Retorne APENAS um objeto JSON no seguinte formato:**
```json
{
  "score": 0.0,
  "rationale": "Explicação breve da pontuação atribuída"
}
```
