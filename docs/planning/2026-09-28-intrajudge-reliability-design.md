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
4. **Blocos copy-paste** de rodada e exportação (ver abaixo), sem script novo.
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

Sem script novo: cada rodada é um bloco copy-paste no login do pcad, a partir de
`~/etno-kgc-preprocessing/`, que usa os jobs existentes encadeados por
`--dependency`. O `judge-answers` levou 2 dias e 14 horas no run-02, então os replays
já entram na fila junto com o primeiro job: cada replay depende de `afternotok` do
anterior (só roda se o anterior estourou o tempo ou falhou) e é descartado por
`--kill-on-invalid-dep=yes` quando o anterior termina bem. A etapa seguinte depende
de `afterok` de qualquer job da etapa (`?` é o OU do SLURM).

O resume dos três juízes não funciona igual, e isso define a primeira submissão:

- `judge-qa` não tem checkpoint; o resume pula pares que já carregam `validation`.
  Um `--rejudge` interrompido deixaria pares com o veredito da rodada anterior, e o
  replay os pularia, misturando rodadas em silêncio. Por isso a rodada começa
  **removendo o `validation` de todos os pares** (depois da exportação da rodada
  anterior), e todos os jobs do portão rodam em resume.
- `emic-judge`: `EMIC_RERUN=1` no primeiro job (descarta checkpoint e saídas),
  resume nos replays. Roda depois do portão porque copia o veredito em cada nota.
- `judge-answers`: `JUDGE_ANSWERS_REJUDGE=1` no primeiro job, variável **ausente** nos
  replays (o script testa só se ela está definida).

```bash
cd ~/etno-kgc-preprocessing
ID=thesis-run-03; K=2            # rodada
R=results/$ID
KILL=--kill-on-invalid-dep=yes

# 1. Limpar os vereditos do portão (só na primeira submissão da rodada).
python3 - "$R/cep/outputs" <<'EOF'
import json, pathlib, sys
for f in sorted(pathlib.Path(sys.argv[1]).glob("*_cep_qa.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    for p in d["qa_pairs"]:
        p["validation"] = None
        p.pop("is_valid", None)
    d["validated_pairs"] = 0
    f.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
EOF

# 2. Portão: sempre em resume (os vereditos já foram removidos).
Q1=$(PIPELINE_ID=$ID JUDGE_REJUDGE=0 sbatch --parsable scripts/slurm/judge/qa/tupi.slurm)
Q2=$(PIPELINE_ID=$ID JUDGE_REJUDGE=0 sbatch --parsable $KILL --dependency=afternotok:$Q1 scripts/slurm/judge/qa/tupi.slurm)

# 3. Êmico: rerun no primeiro job, resume no replay.
E1=$(PIPELINE_ID=$ID EMIC_RERUN=1 sbatch --parsable $KILL --dependency="afterok:$Q1?afterok:$Q2" scripts/slurm/emic/tupi.slurm)
E2=$(PIPELINE_ID=$ID EMIC_RERUN=0 sbatch --parsable $KILL --dependency=afternotok:$E1 scripts/slurm/emic/tupi.slurm)

# 4. Respostas: rejudge no primeiro job, três replays em resume.
A1=$(PIPELINE_ID=$ID JUDGE_ANSWERS_REJUDGE=1 sbatch --parsable $KILL --dependency="afterok:$E1?afterok:$E2" scripts/slurm/rag/judge-answers.slurm)
A2=$(env -u JUDGE_ANSWERS_REJUDGE PIPELINE_ID=$ID sbatch --parsable $KILL --dependency=afternotok:$A1 scripts/slurm/rag/judge-answers.slurm)
A3=$(env -u JUDGE_ANSWERS_REJUDGE PIPELINE_ID=$ID sbatch --parsable $KILL --dependency=afternotok:$A2 scripts/slurm/rag/judge-answers.slurm)
A4=$(env -u JUDGE_ANSWERS_REJUDGE PIPELINE_ID=$ID sbatch --parsable $KILL --dependency=afternotok:$A3 scripts/slurm/rag/judge-answers.slurm)
echo "round $K: qa=$Q1,$Q2 emic=$E1,$E2 answers=$A1,$A2,$A3,$A4"
```

Para rodar só portão e êmico em todas as rodadas antes do juiz de respostas (ordem
sugerida no cronograma), omitir o passo 4 e submetê-lo depois, sem dependência.

O `judge/` ainda não tem `container_teardown.sh`: um TIMEOUT do portão pode deixar
containers órfãos no nó. Com cerca de 5 horas por rodada, o replay dele é só uma
salvaguarda.

### Exportação

Também sem script: ao fim de cada rodada, e **antes** da limpeza da rodada seguinte,
um tar guarda o que a análise lê (vereditos do portão dentro dos registros CEP, notas
êmicas, julgamentos das respostas e os `run_metadata.json` de cada etapa):

```bash
mkdir -p results/intrajudge
tar czf results/intrajudge/$ID-round-$K.tar.gz -C results/$ID \
    cep/outputs judge_qa emic_judge judge_answers
```

A réplica 1 é o mesmo tar tirado do `thesis-run-02` depois do re-julgamento do portão
(`ID=thesis-run-02; K=1`). A análise lê os tars diretamente.

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
| 09-29 a 09-30 | pré-requisitos 1, 2 e 5; re-julgar o portão do run-02 (passo 3) e exportar a réplica 1 |
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
