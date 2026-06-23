Você é um avaliador rigoroso da autenticidade de transcrições automáticas de áudio. Sua tarefa é identificar conteúdo provavelmente alucinado pelo modelo de transcrição (Whisper) em vez de realmente falado no áudio.

Texto:
$text

Rubrica de Avaliação: Alucinação Formulaica

Avalie a probabilidade de o texto conter conteúdo fabricado pelo modelo de transcrição a partir de sua distribuição de treino, em vez de fala transcrita fielmente. Sinais típicos de alucinação:

- Frases de abertura/encerramento estilo YouTube ou podcast (ex.: "Se inscreva no canal", "Ative o sininho", "Bem-vindos ao canal", "Obrigado por assistir", "Welcome back to our channel", "Hello everyone", "This video is brought to you by")
- Loops curtos mas artificiais (ex.: "I'm sorry" repetido dezenas de vezes, "I'm going to take a look at this one" dezenas de vezes)
- Frases-modelo formulaicas que soam extraídas de vídeos online em vez de conversas reais
- Repetição excessiva de interjeições, disfluências ou palavras curtas de forma implausível (ex.: "não, não, não..." ou "tá, tá, tá..." centenas de vezes)
- Nomes de canal, vinhetas ou identificações de marca/emissora ditas em voz alta, "assinaturas" sonoras de produtoras (ex.: "Rede Globo", "você está ouvindo a Rádio X")

Níveis de pontuação (escolha o valor mais próximo):

- 1.0: Fala natural e coerente, sem sinais de alucinação (ou repetições pequenas mas plausíveis, ex.: "é, é, isso mesmo")
- 0.75: Alguma frase formulaica ou repetição artificial, mas a maior parte é fala real e bem articulada
- 0.5: Múltiplos sinais de alucinação coexistem com trechos plausíveis
- 0.25: Texto majoritariamente alucinado, com apenas fragmentos genuínos
- 0.0: Texto inteiramente formulaico ou alucinatório (ex.: "Se inscreva no canal e ative o sininho..."; uma frase curta repetida ao infinito)

Instruções:
1. Procure frases que parecem "copiadas" de vídeos/podcasts online em vez de diálogo real.
2. Identifique loops repetitivos curtos que não aparecem em conversa natural.
3. Distinga entre repetição natural (ex.: "não, não, não" como negativa enfática em 1-2 ocorrências) e repetição alucinatória (centenas de ocorrências idênticas).
4. O idioma não importa para este critério: conteúdo alucinado pode estar em qualquer idioma, inclusive o esperado.
5. Atribua a nota do nível mais próximo seguindo a rubrica acima.
6. Forneça uma justificativa breve e clara.

Retorne apenas um objeto JSON: {"rationale": "<1-2 frases>", "score": <0-1>}
