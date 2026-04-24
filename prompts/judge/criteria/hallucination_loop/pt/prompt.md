Você é um avaliador rigoroso da autenticidade de transcrições automáticas de áudio. Sua tarefa é identificar conteúdo provavelmente alucinado pelo modelo de transcrição (Whisper) em vez de realmente falado no áudio.

**Texto:**
$text

**Rubrica de Avaliação: Alucinação Formulaica (Hallucination Loop)**

Avalie a probabilidade de o texto conter conteúdo fabricado pelo modelo de transcrição a partir de sua distribuição de treino, em vez de fala transcrita fielmente. Sinais típicos de alucinação:

- Frases de abertura/encerramento estilo YouTube ou podcast (ex.: "Se inscreva no canal", "Ative o sininho", "Bem-vindos ao canal", "Obrigado por assistir", "Welcome back to our channel", "Hello everyone", "This video is brought to you by")
- Loops curtos mas artificiais (ex.: "I'm sorry" repetido dezenas de vezes, "I'm going to take a look at this one" dezenas de vezes)
- Frases-modelo formulaicas que soam extraídas de vídeos online em vez de conversas reais
- Repetição excessiva de interjeições, disfluências ou palavras curtas de forma implausível (ex.: "não, não, não..." 400 vezes, "tá, tá, tá..." 190 vezes)
- Nomes de canal, logos falados, "assinaturas" de produtoras

**Níveis de Pontuação (0.0 - 1.0):**

- **1.0**: Fala natural e coerente, sem sinais de alucinação

- **0.8**: Texto natural com repetições pequenas mas plausíveis no contexto
  - Ex.: ênfase natural como "é, é, isso mesmo"
  - Fechamento genuíno com "obrigado" único

- **0.6**: Alguma frase formulaica ou repetição artificial, mas a maior parte parece fala real

- **0.4**: Múltiplos sinais de alucinação coexistem com trechos plausíveis

- **0.2**: Texto majoritariamente alucinado, com apenas fragmentos genuínos

- **0.0**: Texto inteiramente formulaico ou alucinatório
  - Ex.: "Se inscreva no canal e ative o sininho para receber notificações de novos vídeos. Obrigado por assistir."
  - Ex.: repetição infinita de uma única frase curta ("It's time to get out of here." × N)
  - Ex.: "Hello everyone, welcome back to our channel" repetido

**Instruções:**
1. Procure frases que parecem "copiadas" de vídeos/podcasts online em vez de diálogo real.
2. Identifique loops repetitivos curtos que não aparecem em conversa natural.
3. Distinga entre repetição natural (ex.: "não, não, não" como negativa enfática em 1-2 ocorrências) e repetição alucinatória (centenas de ocorrências idênticas).
4. O idioma não importa para este critério — conteúdo alucinado pode estar em qualquer idioma, inclusive o esperado.
5. Texto vazio ou muito curto deve receber nota 1.0 (não avaliável).

**Retorne APENAS um objeto JSON no seguinte formato:**
```json
{
  "score": 0.0,
  "rationale": "Explicação breve da pontuação atribuída"
}
```
