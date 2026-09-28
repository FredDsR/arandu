# Confiabilidade intrajuiz (teste-reteste dos juízes LLM)

**Data:** 2026-09-28
**Prazo de uso:** COLING 2027 (submissão em 2026-10-12); versão completa na dissertação
**Sessão cortex:** `~/.cortex/workspaces/FredDsR-arandu/sessions/intrajudge-reliability/`
**Premissa que operacionaliza:** P5 do paper ("judges are instruments")

## Objetivo

Medir quanto da saída de cada juiz LLM é reprodutível sob configuração idêntica, e
quanto dos números publicados (aprovação do portão, EV, KC, Hall, OC) sobrevive ao
ruído de re-julgamento. O estudo é **confirmatório**: thresholds, prompts e regra
de agregação são fixados antes da primeira réplica e não são ajustados a partir dos
resultados. O próprio processo é o benchmark, então usar estes números para
parametrizar os juízes seria vazamento.

## Pré-especificação (congelada antes da réplica 2)

| Item | Valor |
|---|---|
| Modelo | `qwen3:14b` via Ollama, para os três juízes |
| Temperatura | 0,1 para os três juízes |
| τ do portão | 0,625 (entre as âncoras 0,5 e 0,75) |
| τ_e do juiz êmico | 4, por definição da prosa de ancoragem (4 = preserva o núcleo) |
| τ da abstenção | o do config do critério `abstention` no run-02 |
| Prompts | os do commit em que o run-02 canônico for re-julgado; hash registrado no manifesto |
| Veredito canônico | a réplica 1 (run-02 canônico); voto majoritário e mediana entram só como análise de sensibilidade |
| R | 3 réplicas: run-02 canônico + 2 rodadas em `thesis-run-03` |

## Instrumentos

| Juiz | Unidade | Escala | Itens por réplica |
|---|---|---|---|
| `judge-qa` (portão) | par CEP × critério | 0 a 1 em passos de 0,25 (mapeada para 0..4); veredito binário | 2.670 pares; 4 critérios (2 em Remember) |
| `emic-judge` | par CEP | ordinal 1..5 | 2.670 pares |
| `judge-answers` | AnswerRecord × critério | `abstention`, `passage_coverage`, `answer_correctness`, `answer_faithfulness` | 15.020 registros (5 braços × 3.004 probes) |

O `judge-answers` roda sobre todos os probes (aprovados ou não pelo portão e os 334
não respondíveis); `build_gold_lookup` não filtra por aprovação. Ele é, portanto,
independente do veredito do portão. As células TC/FC/FA/TA dependem do critério
`abstention` (`shared/rag/analysis/classifier.py:71`), de modo que o ruído desse juiz
entra também em Hall, OC e F1_abs.

O respondedor (T = 0,2) não é um juiz. As respostas ficam congeladas e o estudo mede
só os juízes.

## Pré-requisitos no código

1. **`judge-qa` grava `run_metadata`** com modelo, provider, temperatura, idioma,
   thresholds e hash dos prompts. Hoje nada registra a temperatura do portão: o
   default do código é 0,1 (`qa/config.py:264`), o compose e o SLURM exportam 0,3
   (`docker-compose.yml:310`, `scripts/slurm/judge/judge_common.sh:35`) e o paper
   declara 0,7 (que é a temperatura do gerador).
2. **Padronizar T = 0,1** no compose, em `judge_common.sh` e no `.env.example`.
3. **Re-julgar o portão do `thesis-run-02`** com o código acima. Esse passa a ser o
   canônico e a réplica 1 do portão.
4. **Exportador de rodada** (ver abaixo).
5. **Script de análise** reutilizando `shared/agreement/coefficients.py`
   (`krippendorff_alpha`, `gwet_ac2`, `cohen_kappa_weighted`, com escala fixa).

Réplica 1 dos outros juízes: o `emic-judge` do run-02 (2026-09-23, T = 0,1, instrumento
atual) e o `judge-answers` do run-02 (2026-07-04, T = 0,1). Desde então os prompts e
o código do juiz de respostas não mudaram (só o rename `TranscriptionRecord` no
resolver). A versão do Ollama e da imagem pode ter mudado, e fica registrada como ameaça.

## Execução das rodadas

`thesis-run-03` é criado uma vez com `arandu replicate thesis-run-02 --id thesis-run-03`,
depois do passo 3. Chunk, KG, non_answerable, retrieve e answers ficam congelados
(os seeds dos probes vêm do portão antigo e não devem ser regenerados). Em cada rodada
k só os três juízes rodam, e as saídas são exportadas antes da rodada seguinte.

### Encadeamento no SLURM com replay

Um job único por rodada (`scripts/slurm/intrajudge/round.slurm`) executa as etapas
em sequência: `judge-qa` → `emic-judge` → `judge-answers` → exportação. O
`judge-answers` levou 2 dias e 14 horas no run-02, então o job passa do teto de 24h
e precisa de replay. O job é idempotente e guiado por um arquivo de estado
`results/thesis-run-03/intrajudge/round-<k>/state.json`:

- Cada etapa tem os estados `pending`, `started` e `done`.
- Na **primeira** entrada de uma etapa, o job limpa o resultado anterior e marca
  `started`. Nos replays ele entra em modo resume.
- A limpeza é específica para cada juiz, porque o resume dos três não funciona igual:
  - `judge-qa` não tem checkpoint; o resume pula pares que já carregam `validation`.
    Um `--rejudge` interrompido deixaria pares com o veredito da rodada anterior e o
    replay os pularia, misturando rodadas em silêncio. Por isso a primeira entrada
    **remove o campo `validation` de todos os pares** (depois da exportação da
    rodada anterior) e roda sempre em resume.
  - `emic-judge`: `--rerun` na primeira entrada (descarta o checkpoint), `--resume`
    nos replays.
  - `judge-answers`: `--rejudge` na primeira entrada, resume nos replays.
- `emic-judge` roda depois de `judge-qa` porque copia o veredito do portão em cada nota.
- A etapa só vira `done` com exit 0 e checagem de completude (número de itens julgados
  igual ao esperado, falhas contadas à parte).
- Replay: `ROUND=<k> sbatch scripts/slurm/intrajudge/round.slurm`, o mesmo comando
  até o estado final. `STAGES` permite restringir as etapas (ver ordem sugerida).
- Segue `container_teardown.sh` e `#SBATCH --signal=B:TERM@60`, como `rag/` e `emic/`.

### Exportação

`results/thesis-run-03/intrajudge/round-<k>/` guarda apenas o necessário para a
análise, em formato compacto:

- `gate.jsonl`: `qa_pair_id`, nível Bloom, nota por critério, `passed`, `rejected_at`,
  erro de parse;
- `emic.jsonl`: `qa_pair_id`, nota, erro;
- `answers.jsonl`: braço, `qa_pair_id`, `is_answerable`, `abstained` do respondedor,
  nota por critério;
- `manifest.json`: modelo, digest da imagem, versão do Ollama, temperatura,
  thresholds, hash dos prompts, commit, horários, contagens.

A réplica 1 é exportada do `thesis-run-02` no mesmo formato.

## Métricas

1. **Coeficientes de confiabilidade**, com as réplicas tratadas como codificadores:
   - α ordinal para cada critério do portão (escala 0..4), para o EV (1..5) e para
     cada critério do juiz de respostas;
   - α nominal para o veredito do portão e para a decisão de abstenção;
   - Gwet AC2 como verificação de robustez contra o paradoxo de prevalência
     (Remember aprova 94,6%; EV = 5 em 57,9%);
   - leitura pelos limiares de Krippendorff: α ≥ 0,800 é confiável, 0,667 ≤ α < 0,800
     só permite conclusões provisórias.
   - Tudo estratificado por nível de Bloom (e por braço no juiz de respostas).
2. **Instabilidade por item.** π̂ᵢ = kᵢ/R e Dᵢ = 2kᵢ(R−kᵢ)/(R(R−1)). Os itens se dividem
   em núcleo estável, faixa instável e rejeitados estáveis. No ordinal, reportar a
   amplitude e a fração de itens com amplitude ≥ 2.
3. **Viés versus ruído.** Δ médio entre réplicas (esperado nulo) contra |Δ| médio;
   tendência ao longo das rodadas.
4. **Dependência da margem.** Instabilidade em função da distância entre a nota média
   do item e τ.
5. **Estabilidade dos números publicados.** Média e amplitude entre réplicas das
   tabelas do paper (aprovação por nível, pares úteis, EV por nível, KC/Hall/OC/F1_abs
   por braço) e a preservação do ranking dos braços. Uma diferença entre braços só é
   reportada como achado se o IC do bootstrap pareado (itens × réplicas) excluir zero.
6. **Teto para o estudo humano.** O α intrajuiz do juiz êmico limita por cima o α
   juiz × anotadores.

## Cronograma sugerido

| Data | Passo |
|---|---|
| 09-29 a 09-30 | pré-requisitos 1, 2, 4 e 5; re-julgar o portão do run-02 (passo 3) |
| 10-01 | `replicate` → `thesis-run-03`; rodadas 2 e 3 com `STAGES=judge-qa,emic-judge` (cerca de 8h cada) |
| 10-02 a 10-08 | rodadas 2 e 3 do `judge-answers` (cerca de 62h cada, com replays) |
| 10-08 a 10-10 | análise e texto do paper |

Rodar primeiro o portão e o êmico em todas as rodadas deixa a parte do paper que
depende deles pronta cedo, mesmo que o juiz de respostas atrase.

## Riscos e efeitos colaterais

- **Re-julgar o portão do run-02 muda números do paper**: tabela do portão,
  caracterização do benchmark, coortes, cruzamento êmico × aprovação.
- **A amostra humana (120 pares) foi sorteada entre os 1.612 aprovados pelo portão
  antigo.** Alguns podem passar a ser reprovados. A amostra é mantida (ela valida o EV,
  não o portão), mas isso precisa ser dito.
- **O veredito copiado nas notas êmicas do run-02 fica obsoleto**. A análise deve
  fazer o join com o veredito atual, sem re-pontuar.
- **Prazo.** Se o juiz de respostas não fechar a rodada 3, o paper reporta R = 2 para
  ele e R = 3 para portão e êmico.
- **Paralelismo.** As rodadas são sequenciais porque compartilham `thesis-run-03`.
