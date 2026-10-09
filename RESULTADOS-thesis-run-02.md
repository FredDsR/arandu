# Resultados canônicos: `thesis-run-02`

Documento de referência dos resultados do pipeline. Consolidado em 2026-09-21 a
partir do arquivo publicado no Google Drive. Cada número traz o comando que o
produz, para que a revisão possa refazer qualquer linha sem reconstruir o
raciocínio.

## 0. Declaração de canonicidade

**`thesis-run-02` é o run canônico. Todo artigo, capítulo ou dissertação escrito
a partir de agora deve citar os números deste documento.**

O `thesis-run-01` passa a ser histórico. Ele foi julgado com um juiz que
recebia a transcrição inteira como contexto de fundamentação em vez do chunk que
originou o par, corrigido em `38f58a28` (#168). O `thesis-run-02` é o mesmo
corpus com o estágio `cep` re-julgado sob o contexto correto, mais três estágios
que o run anterior não tinha (`emic_judge`, `human_eval`, `annotation`).

Números que mudam de um run para o outro e que precisam ser conferidos em
qualquer texto já escrito:

| Grandeza | thesis-run-01 (obsoleto) | **thesis-run-02 (canônico)** |
|---|---|---|
| pares aprovados pelo portão CEP (base 2670) | 1579 | **1612** |
| pares aprovados (base 2652, com piso de chunk) | 1585 | **1604** |
| pares úteis (aprovados acima de `remember`) | 354 | **349** |
| pares sem nota por falha do juiz | 12 | **1** |
| chunks aproveitados | 242 de 442 | **248 de 442** |
| documentos com ao menos um par útil | 128 de 214 | **129 de 214** |

As tabelas do benchmark RAG (seção 4.5) são idênticas nos dois runs, porque
apenas o estágio `cep` foi re-julgado.

**O instrumento de validade êmica mudou em 2026-09-23.** O commit `764eac15`
(#176) passou a mostrar ao juiz êmico e ao anotador humano o bloco de metadados
da entrevista que a geração CEP sempre injetou. É mudança de instrumento, não de
dado: notas êmicas anteriores a essa data não são comparáveis às atuais, e
anotação humana coletada com o instrumento cego não pode ser confrontada com um
juiz que enxerga os metadados. Os números da seção 4.7 são os do instrumento
vigente; a seção 4.7.1 mede o efeito da mudança.

## 1. Procedência do dado

A pasta do Drive é
`https://drive.google.com/drive/folders/1OAELvX2akMaKJKheh5NWfHrH0jzIsjAp` e o
arquivo `thesis-run-02.tgz` mantém sempre o mesmo id,
`1qJUuzIUYbvPbreCm-mfNEDz1ra8BK_hC`. Três versões dele entram neste documento:

| Versão | tamanho (bytes) | md5 | o que traz |
|---|---|---|---|
| **2026-09-24, segunda (vigente)** | 227 296 588 | `3903a6385ff82871527ea9efbde42b53` | juiz êmico com metadados, instrumento humano reconstruído, `analysis` regerado |
| 2026-09-24, primeira (substituída) | 227 515 814 | `4ed668f52a36adc8e85945d885e333f5` | juiz êmico com metadados, mas `human_eval` e `annotation` ainda cegos |
| 2026-09-21 | 227 517 922 | `050e751e53fa4f9bde6eb219836deb15` | juiz êmico cego, linha de base da seção 4.7.1 |

Entre a versão de 2026-09-21 e as de 2026-09-24 mudaram apenas os estágios
`emic_judge` (os 214 arquivos), `human_eval`, `annotation` e `analysis`. `cep`,
`chunk`, `transcription`, `non_answerable`, `retrieve`, `answers` e
`judge_answers` são byte a byte idênticos, então tudo das seções 4.1 a 4.6 e da
seção 5 permanece válido sem re-execução.

A versão com índices, `thesis-run-02_with-indexes.tgz` (id
`1T-5kbaoXyN-k5igIOvVL5Bcz_I88WpkC`), foi republicada junto: 726 483 882 bytes,
md5 `606deaaa800118a2a8c8390963f141a0`.

Download e extração:

```bash
curl -L -o ~/Downloads/thesis-run-02.tgz \
  "https://drive.usercontent.google.com/download?id=1qJUuzIUYbvPbreCm-mfNEDz1ra8BK_hC&export=download&confirm=t"
md5sum ~/Downloads/thesis-run-02.tgz   # 3903a6385ff82871527ea9efbde42b53

mkdir -p results/thesis-run-02
tar -xzf ~/Downloads/thesis-run-02.tgz -C results/thesis-run-02
uv run arandu rebuild-index
```

O tarball tem raiz `./`, então a extração precisa do diretório criado antes.

**Índices de recuperação.** O tarball enxuto não traz os índices pesados: ele
mantém os dois `manifest.json` e deixa de fora `retrieve/indexes/bm25_cep_4k/`
`bm25.pkl`, os dez arquivos de `kg/outputs/atlas_output/precompute/` e o
`transcriptions.json_without_concept.pkl`. Para tê-los, baixe
`thesis-run-02_with-indexes.tgz` e extraia por cima, ou extraia só ele. Desde a
republicação de 2026-09-24 os dois arquivos saem da mesma árvore, então a
procedência é integralmente do Drive. Os índices só são necessários para
re-executar `arandu retrieve`; nenhum resultado deste documento depende deles.

### 1.1 Republicação de 2026-09-24

A primeira versão publicada naquele dia trazia o juiz êmico novo mas ainda os
`human_eval` e `annotation` cegos, defasados em relação à árvore local. Os dois
tarballs foram regerados da árvore completa e republicados no mesmo dia, nos
mesmos ids do Drive.

| Arquivo | tamanho (bytes) | md5 |
|---|---|---|
| `thesis-run-02.tgz` | 227 296 588 | `3903a6385ff82871527ea9efbde42b53` |
| `thesis-run-02_with-indexes.tgz` | 726 483 882 | `606deaaa800118a2a8c8390963f141a0` |

**Conferido:** os dois arquivos foram baixados do Drive depois do upload, em
2026-09-24 13:38, e os md5 batem com os da tabela. O que está publicado é
exatamente a árvore descrita neste documento.

Como foram gerados, com a mesma convenção dos anteriores (raiz `./`, e a versão
enxuta sem os índices pesados, mantendo os `manifest.json`):

```bash
OUT=~/Downloads/thesis-run-02-upload-2026-09-24
mkdir -p $OUT

tar -czf $OUT/thesis-run-02.tgz -C results/thesis-run-02 \
  --exclude='./retrieve/indexes/bm25_cep_4k/bm25.pkl' \
  --exclude='./kg/outputs/atlas_output/kg_graphml/transcriptions.json_without_concept.pkl' \
  --exclude='./kg/outputs/atlas_output/precompute/*.pkl' \
  --exclude='./kg/outputs/atlas_output/precompute/*.index' \
  .

tar -czf $OUT/thesis-run-02_with-indexes.tgz -C results/thesis-run-02 .
```

Conferências feitas nos arquivos gerados: 46 607 entradas no enxuto e 46 619 no
com-índices (exatamente os 12 arquivos pesados a mais), nenhum dos 12 presente
no enxuto, os dois `manifest.json` preservados, e a extração de amostra
confirmando `tables.md` com cabeçalho `thesis-run-02`, `emic_judge` de
2026-09-23 e `human_eval` com o campo `metadata`.

## 2. Ambiente

```bash
uv sync --all-extras --all-groups
uv run arandu --version
```

Todos os comandos deste documento rodam da raiz do repositório. Os scripts de
`scripts/` precisam ser chamados como módulo (`python -m scripts.<nome>`), senão
o import de `scripts._judge_analysis_common` falha.

## 3. Mapa de artefatos

| Caminho | Conteúdo |
|---|---|
| `results/thesis-run-02/cep/outputs/*_cep_qa.json` | 214 arquivos, um por entrevista, com os 2670 pares e o veredito do juiz |
| `results/thesis-run-02/judge_answers/outputs/<braço>/{cep,nonanswerable}/` | resposta e julgamento de cada par em cada braço |
| `results/thesis-run-02/analysis/outputs/tables.md` | tabelas do benchmark RAG |
| `results/thesis-run-02/analysis/outputs/report.json` | mesmas métricas em JSON, com matrizes de confusão e IC |
| `results/thesis-run-02/analysis/outputs/emic_scores.md` | análise descritiva do juiz de validade êmica |
| `results/thesis-run-02/emic_judge/outputs/*.json` | nota êmica por par |
| `results/thesis-run-02/human_eval/outputs/sample.jsonl` | amostra de 120 pares para anotação humana |
| `results/thesis-run-02/annotation/outputs/tasks.json` | as mesmas 120 tarefas em formato Label Studio |
| `results/thesis-run-02/analysis/kg_structure_profile.json` | perfil estrutural do grafo (seção 4.9) |
| `notebooks/bloom-thesis-run-02.ipynb` | notebook do funil CEP executado sobre o run-02 |
| `notebooks/figures/*-thesis-run-02.{pdf,png}` | figuras do notebook, em versão de artigo |

O `notebooks/bloom.ipynb` continua apontando para o `thesis-run-01` e guarda as
saídas antigas. Ele é histórico; não use seus números.

**Duas bases de contagem.** Os comandos `arandu` e os scripts de `scripts/`
contam sobre os **2670 pares** do corpus bruto. O notebook aplica um piso de 100
caracteres de chunk, que descarta 3 chunks (18 pares) que são sobras da última
janela do chunker, e conta sobre **2652 pares / 442 chunks**. As duas bases estão
corretas; o que não pode é misturá-las na mesma frase.

## 4. Resultados

### 4.1 Corpus e chunking

| Métrica | Valor |
|---|---|
| documentos (entrevistas) | 214 |
| chunks gerados (`cep_4k`) | 445 |
| pares QA gerados | 2670 |
| extensão dos documentos (mediana) | 2950 caracteres |
| extensão dos documentos (máxima) | 134 971 caracteres |
| razão maior/menor documento | 628x |
| chunks por documento (mediana) | 1 |
| documentos de um único chunk | 136 de 214 (63,6%) |
| subsídio de chunks ao quartil mais curto | 6,93x o proporcional ao texto |

*Como reproduzir:* execute o notebook e leia as saídas das células 3 e 27.

```bash
cd notebooks && uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 bloom-thesis-run-02.ipynb
```

O notebook precisa rodar com `notebooks/` como diretório de trabalho: é de lá que
ele resolve a raiz do repositório e grava o export manual de pares.

### 4.2 Portão CEP

Aprovação marginal por critério e nível (base 2652):

| Critério | remember | understand | analyze | evaluate |
|---|---|---|---|---|
| Fidelidade | 94,7% | 89,6% | 81,2% | 78,1% |
| Calibração Bloom | 99,9% | 95,0% | 95,5% | 77,6% |
| Informatividade | não avaliado | 56,3% | 62,4% | 49,1% |
| Autocontenção | não avaliado | 48,0% | 65,8% | 58,6% |

`informativeness` e `self_containedness` não são aplicados a `remember` por
configuração do juiz, então célula vazia é ausência de avaliação, não reprovação.

Funil:

| Métrica | Valor |
|---|---|
| aprovados em todos os critérios avaliados | 1604 de 2652 (60,5%) |
| `is_valid` gravado pelo pipeline | 1603 (60,4%) |
| pares sem nota (falha do juiz) | 1 |
| candidatos a par útil (níveis acima de `remember`) | 1326 (50,0% dos gerados) |
| **pares úteis** | **349 (26,3% dos candidatos, 13,2% dos gerados)** |
| por nível | `understand` 105 (23,8%), `analyze` 154 (34,8%), `evaluate` 90 (20,4%) |

Decomposição da reprovação, em % dos pares gerados no nível:

| Nível | aprovado | só Fidel. | só Calib. | só Inform. | só Autocont. | 2+ critérios |
|---|---|---|---|---|---|---|
| remember | 94,6 | 5,3 | 0,1 | 0,0 | 0,0 | 0,0 |
| understand | 23,8 | 1,1 | 0,9 | 19,5 | 25,8 | 29,0 |
| analyze | 34,8 | 5,9 | 0,9 | 18,8 | 15,6 | 24,0 |
| evaluate | 20,4 | 3,4 | 5,7 | 17,4 | 11,5 | 41,6 |

Cenários de remoção de critério (total aprovado, inclui `remember`):

| Configuração | aprovados | fração do corpus |
|---|---|---|
| atual (4 critérios) | 1604 | 60,5% |
| sem fidelidade | 1720 | 64,9% |
| sem calibração Bloom | 1638 | 61,8% |
| sem informatividade | 1850 | 69,8% |
| sem autocontenção | 1838 | 69,3% |

*Como reproduzir:* notebook, células 8 (aprovação marginal), 12 (funil), 13
(cenários) e 16 (decomposição). A contagem equivalente sobre os 2670 pares sai de:

```bash
uv run python -m scripts.analyze_judge_thresholds --id thesis-run-02
```

### 4.3 Rendimento por chunk e por documento

| Métrica | Valor |
|---|---|
| chunks aproveitados (>= 1 par útil) | 248 de 442 (56,1%) |
| pares úteis por chunk (geral) | 0,79 |
| pares úteis por chunk aproveitado | 1,41 (teto de 3) |
| chunks com os 3 pares úteis | 15 (3,4%) |
| documentos com >= 1 par útil | 129 de 214 (60,3%) |

Por faixa de tamanho de chunk:

| Faixa | chunks | taxa de úteis | úteis por 1k caracteres |
|---|---|---|---|
| <0,5k | 27 | 4,9% | 0,49 |
| 0,5-1k | 35 | 20,0% | 0,85 |
| 1-2k | 75 | 28,4% | 0,57 |
| 2-3k | 51 | 26,8% | 0,31 |
| 3-3,5k | 39 | 27,4% | 0,25 |
| 3,5-3,9k | 87 | 27,2% | 0,22 |
| 3,9-4k | 128 | 30,2% | 0,23 |

Spearman entre tamanho do chunk e taxa de úteis: +0,144. A taxa sobe com o
tamanho e a densidade cai, porque o orçamento de 6 pares por chunk é fixo.

Concentração:

| Métrica | Valor |
|---|---|
| Gini dos caracteres por documento | 0,621 |
| Gini dos candidatos (após chunking) | 0,418 |
| Gini dos pares úteis | 0,681 |
| top 5% dos documentos | 37,5% dos pares úteis |
| top 25% dos documentos | 74,2% dos pares úteis |
| documentos sem nenhum par útil | 85 de 214 |

*Como reproduzir:* notebook, células 20 a 27. As figuras correspondentes são
`notebooks/figures/rendimento-chunk-thesis-run-02.pdf` e
`cobertura-corpus-thesis-run-02.pdf`.

### 4.4 Distribuição das notas do juiz (base 2670)

| Critério | n | média | abaixo de 0,625 | 0,0 | 0,25 | 0,5 | 0,75 | 1,0 |
|---|---|---|---|---|---|---|---|---|
| fidelidade | 2669 | 0,860 | 11,2% | 18 | 12 | 269 | 844 | 1526 |
| calibração Bloom | 2670 | 0,945 | 5,5% | 1 | 66 | 79 | 231 | 2293 |
| informatividade | 1335 | 0,648 | 44,4% | 0 | 101 | 492 | 592 | 150 |
| autocontenção | 1335 | 0,695 | 42,6% | 44 | 55 | 470 | 349 | 417 |

As notas são quantizadas em 5 âncoras e o limiar 0,625 cai no vão entre 0,5 e
0,75. O portão é insensível entre 0,625 e 0,7 (1612 aprovados nos dois); em 0,5
sobe para 2394.

*Como reproduzir:*

```bash
uv run python -m scripts.analyze_judge_thresholds --id thesis-run-02
uv run python -m scripts.plot_judge_score_distributions --id thesis-run-02 --style bars
uv run python -m scripts.plot_judge_score_distributions --id thesis-run-02 --style density
uv run python -m scripts.plot_judge_score_distributions --id thesis-run-02 --style hist
```

As figuras vão para `results/thesis-run-02/analysis/judge_score_*.png`.

### 4.5 Benchmark RAG

Todos os 2670 pares têm recuperação, resposta e julgamento nos 5 braços, mais
334 itens não respondíveis.

| Braço | KC | Alucinação | Cautela excessiva | F1 de abstenção | Cob. de passagem | Recuperação da fonte |
|---|---|---|---|---|---|---|
| atlas_rag_hipporag | 0,592 | 0,198 | 0,587 | 0,247 | 0,528 | 0,077 |
| bm25_cep_4k | 0,613 | 0,365 | 0,325 | 0,300 | 0,736 | 0,139 |
| khop_passage | 0,585 | 0,216 | 0,613 | 0,235 | 0,606 | 0,078 |
| khop_triple | 0,464 | 0,054 | 0,857 | 0,215 | 0,239 | n/a |
| null | n/a | 0,000 | 1,000 | 0,200 | 0,000 | n/a |

KC por nível de Bloom:

| Braço | remember | understand | analyze | evaluate |
|---|---|---|---|---|
| atlas_rag_hipporag | 0,553 | 0,643 | 0,597 | 0,645 |
| bm25_cep_4k | 0,576 | 0,654 | 0,665 | 0,641 |
| khop_passage | 0,543 | 0,656 | 0,630 | 0,610 |
| khop_triple | 0,467 | 0,468 | 0,452 | 0,440 |

KC por tipo de pergunta:

| Braço | conceitual | factual |
|---|---|---|
| atlas_rag_hipporag | 0,629 | 0,553 |
| bm25_cep_4k | 0,654 | 0,576 |
| khop_passage | 0,637 | 0,543 |
| khop_triple | 0,458 | 0,467 |

*Como reproduzir:*

```bash
uv run arandu rag-analysis --id thesis-run-02
cat results/thesis-run-02/analysis/outputs/tables.md
```

O comando relê `judge_answers/outputs/` e cruza com `cep/outputs/` para as
estratificações. Ele não chama LLM nenhuma: é agregação pura.

### 4.6 Coorte admitida apenas no limiar 0,5

| Conjunto | n | KC (bm25) | Correção (bm25) |
|---|---|---|---|
| aprovados (>= 0,625) | 1612 | 0,604 | 0,644 |
| coorte 0,5 (todos) | 782 | 0,631 | 0,668 |
| 0,5 só por autocontenção | 195 | 0,629 | 0,668 |
| 0,5 só por informatividade | 207 | 0,688 | 0,731 |

Os pares excluídos no limiar pontuam **acima** dos aprovados no desfecho RAG.
Isso não indica erro do portão: informatividade e autocontenção medem valor de
conhecimento e independência de contexto, não respondibilidade, e perguntas
genéricas são mais fáceis de responder.

*Como reproduzir:*

```bash
uv run python -m scripts.analyze_qa_cohort_rag_outcome --id thesis-run-02
```

### 4.7 Juiz de validade êmica

Instrumento vigente: juiz com os metadados da fonte (commit `764eac15`, #176),
executado no cluster em 2026-09-23, `qwen3:14b` a temperatura 0,1, tomando
`results/thesis-run-02/cep/outputs` como entrada.

| Métrica | Valor |
|---|---|
| pares pontuados | 2670 de 2670 (nenhum erro de LLM) |
| média | 4,14 |
| mediana | 5,00 |
| nota >= 4 | 60,1% |
| distribuição | 1: 0,9% · 2: 2,7% · 3: 36,4% · 4: 2,2% · 5: 57,9% |

Por nível de Bloom:

| Nível | n | média | >= 4 |
|---|---|---|---|
| remember | 1335 | 4,69 | 86,7% |
| understand | 445 | 3,92 | 49,7% |
| analyze | 445 | 3,45 | 26,5% |
| evaluate | 445 | 3,38 | 24,0% |

Cruzamento com o veredito do portão CEP, sobre o portão vigente:

| Veredito | n | média | >= 4 |
|---|---|---|---|
| aprovado | 1612 | 4,52 | 78,3% |
| reprovado | 1058 | 3,55 | 32,3% |

A escala continua sendo usada de forma bimodal (36,4% em 3 e 57,9% em 5) e a
nota continua caindo monotonicamente ao subir a escada de Bloom, o inverso do
que a hipótese de elicitação de conhecimento tácito prevê. A separação entre
aprovados e reprovados do portão CEP, em compensação, ficou nítida (4,52 contra
3,55), o que é evidência a favor do portão.

*Como reproduzir:*

```bash
uv run python scripts/analyze_emic_scores.py --id thesis-run-02 \
  --out results/thesis-run-02/analysis/outputs/emic_scores.md
```

Este script roda direto (não como módulo) porque não importa
`scripts._judge_analysis_common`.

### 4.7.1 Efeito da injeção de metadados no juiz êmico

O #176 parte de uma previsão específica: o item 3 da escala êmica ("acrescenta
algo que a pessoa não disse") estava disparando em pares que citam um valor de
metadado que o juiz não podia ver, então **apenas** esses pares deveriam subir
ao receber o bloco. O teste é uma diferença em diferenças entre o run cego
(2026-08-19) e o run com metadados (2026-09-23), sobre os mesmos pares, o mesmo
modelo e a mesma temperatura.

O grupo tratado é definido por regra explícita: o par cita, na pergunta ou na
resposta, um valor de metadado da entrevista (participante, pesquisador, local,
data, contexto) que **não** aparece no próprio chunk. São 456 dos 2670 pares
(17,1%). A regra é um casamento de substring com normalização de caixa e
acentos, ou seja, uma operacionalização de "cita o metadado", não a leitura do
próprio juiz.

| Grupo | n | média cego | média com metadados | Δ | >= 4 cego | >= 4 com metadados |
|---|---|---|---|---|---|---|
| cita metadado ausente do chunk | 456 | 3,62 | 4,19 | **+0,56** | 35,1% | 63,8% |
| não cita | 2213 | 4,05 | 4,12 | +0,08 | 55,5% | 59,3% |

**Diferença em diferenças: +0,487, IC 95% bootstrap [+0,368, +0,607]** (5000
reamostragens, semente 20260924).

Movimento por par:

| Grupo | subiu | desceu |
|---|---|---|
| cita metadado ausente | 170 (37,3%) | 41 (9,0%) |
| não cita | 328 (14,8%) | 241 (10,9%) |

O grupo que não cita metadado é o piso de ruído desta medida: 14,8% de subida
contra 10,9% de descida é churn de re-julgamento, simétrico e sem direção. O
grupo tratado sobe quatro vezes mais do que desce. O efeito é do instrumento,
não do acaso.

Por nível de Bloom:

| Nível | Δ no nível inteiro | n que citam | Δ entre os que citam |
|---|---|---|---|
| remember | +0,25 | 260 | **+0,97** |
| understand | +0,04 | 64 | +0,06 |
| analyze | +0,06 | 68 | -0,12 |
| evaluate | +0,11 | 64 | +0,14 |

**Leitura.** O fix funcionou exatamente onde foi desenhado para funcionar, e
esse é o contraste com o fix de contexto do juiz CEP da seção 5, cujo efeito era
indistinguível de ruído. Mas o ganho está quase todo em `remember`: dos 456
pares tratados, 260 são de recall factual, e é lá que a nota sobe quase um ponto
inteiro. Nos níveis táticos o subgrupo tratado é pequeno (64 a 68 pares) e o
efeito é nulo. Em termos de tese: a injeção de metadados corrige um falso
negativo real de medição, e não melhora a validade êmica dos pares que
interessam à pergunta de pesquisa.

*Como reproduzir:*

```bash
# a linha de base cega vem do arquivo de 2026-09-21
mkdir -p ~/Downloads/thesis-run-02-emic-blind
tar -xzf ~/Downloads/thesis-run-02.2026-09-21.tgz \
  -C ~/Downloads/thesis-run-02-emic-blind ./emic_judge
mv ~/Downloads/thesis-run-02-emic-blind/emic_judge/* ~/Downloads/thesis-run-02-emic-blind/

uv run python -m scripts.analyze_emic_metadata_effect \
  --id thesis-run-02 --baseline ~/Downloads/thesis-run-02-emic-blind
```

### 4.8 Amostras para anotação humana

Reconstruídas com o instrumento novo: o bloco de metadados chega ao anotador
pelo mesmo portão que alimenta o juiz, e o sorteio foi refeito sobre o portão
CEP vigente.

| Artefato | Conteúdo |
|---|---|
| `human_eval/outputs/sample.jsonl` | 120 itens, 30 por célula de Bloom, seed 42, cada item com o campo `metadata` |
| `human_eval/outputs/sample_manifest.json` | `pipeline_id: thesis-run-02`, população por célula 1263/105/154/90, 1058 excluídos por reprovação, `pool_sha256` `47d108fa230f...` |
| `annotation/outputs/tasks.json` | as mesmas 120 tarefas em Label Studio, seed 24, campo `metadata` por tarefa, projeto 7 |
| `annotation/outputs/labeling_config.xml`, `expert_instruction.html` | interface e instrução, com a passagem `provisions.source_metadata` da régua |

Estes dois estágios chegaram a divergir entre o Drive e a árvore local: o
primeiro arquivo publicado em 2026-09-24 ainda trazia a versão cega
(`pipeline_id: thesis-run-01`, sem campo `metadata`, população 1225/115/146/93,
`pool_sha256` `21c22b9d1918...`). A republicação do mesmo dia (seção 1.1)
resolveu isso, e o arquivo vigente traz a versão acima.

### 4.9 Estrutura do grafo

Perfil do grafo de `kg/outputs/atlas_output/kg_graphml/transcriptions.json_graph.graphml`,
computado por `scripts/kg_structure_profile.py` (seed 0). O subgrafo de relações
é o grafo não-direcionado formado só pelas arestas `Relation` (inclui as arestas
de participação sintetizadas pelo atlas-rag, `envolve`, 34,7% delas); a
procedência de um nó é a passagem a que ele tem aresta `Source` (quase sempre
uma só), e o local vem da linha `Local:` do bloco de metadados da passagem.

| Métrica | Valor |
|---|---|
| nós / arestas (grafo completo) | 18.454 / 76.919 |
| arestas por tipo | Relation 13.928; Concept 52.438; Source 10.553 |
| subgrafo de relações: nós / arestas | 10.586 / 13.748 |
| componentes conexos (subgrafo de relações) | 1.714; maior com 61,0% dos nós |
| grau no subgrafo de relações | mediana 1; p90 3; p99 29; máx. 732 |
| nós com grau 1 | 75,5% |
| inclinação log-log da CCDF (grau >= 3) | -1,24 |
| clustering médio / transitividade | 0,052 / 0,018 |
| 20 maiores hubs tocam | 32,5% das arestas de relação |
| 10 maiores hubs | Dona Gilda (732), Célia (527), a gente (486), Aida (274), água (250), pessoas (234), Henrique (232), comunidade (215), enchente (167), Mamá (164) |
| arestas de relação entre nós da mesma passagem / de passagens distintas | 7.181 / 6.328 (239 sem procedência) |
| arestas de relação entre locais distintos | 12,4% |
| Louvain no componente gigante | 35 comunidades; modularidade 0,629; pureza de local mediana 0,73; 10 com pureza >= 0,8 |
| conceitos: 20 mais conectados concentram | 59,5% das pontas de arestas Concept |
| 10 conceitos mais conectados | evento (4.714), veículo, meio de transporte, categoria, fenômeno, objeto, ação, tipo, manifestação, atividade |
| ego-grafo de 2 saltos a partir de entidade aleatória (n = 300) | mediana 222 nós; p90 2.891; máx. 5.054 |
| idem, nós alcançados só via conceitos | mediana 5; p90 2.777 |
| idem, passagens alcançadas | mediana 2; p90 18 (de 302) |
| rótulos de entidade que diferem só em caixa/espaço | 334 grupos, 673 nós |

Leitura: o subgrafo de relações é uma floresta de estrelas cujos hubs são os
nomes dos entrevistados, pronomes dêiticos e substantivos genéricos; a camada de
conceitos colapsa em tipos vazios; e o conhecimento relacional fica em grande
parte dentro de cada local. O ego-grafo de 2 saltos, que é o que os braços
`khop_*` expandem (sobre todos os tipos de aresta, `_khop_common.py`), explode
quando o primeiro salto cai num conceito genérico.

*Como reproduzir:*

```bash
uv run python -m scripts.kg_structure_profile --id thesis-run-02
```

## 5. Estabilidade do juiz (thesis-run-01 vs thesis-run-02)

Esta seção não é comparação de desempenho: é uma medida de quanto do veredito do
juiz é reprodutível. Como o re-julgamento só reescreve `validation` e
`is_valid`, e o benchmark RAG cobre todos os 2670 pares, a comparação é um
refiltro puro, sem nada a re-executar.

Concordância entre os dois julgamentos:

| Medida | Valor |
|---|---|
| concordância bruta do veredito | 84,1% |
| kappa de Cohen | 0,669 |

Estabilidade por critério:

| Critério | n | mesma âncora | \|Δ\| médio | Δ médio |
|---|---|---|---|---|
| fidelidade | 2658 | 69,6% | 0,097 | +0,001 |
| calibração Bloom | 2670 | 81,8% | 0,059 | -0,001 |
| informatividade | 1335 | 52,7% | 0,132 | -0,016 |
| autocontenção | 1335 | 56,3% | 0,130 | -0,003 |

Coortes de troca de lado:

| Coorte | total | remember | understand | analyze | evaluate |
|---|---|---|---|---|---|
| estável aprovado | 1383 | 1183 | 62 | 95 | 43 |
| novo aprovado | 229 | 80 | 43 | 59 | 47 |
| novo reprovado | 196 | 42 | 53 | 51 | 50 |
| estável reprovado | 862 | 30 | 287 | 240 | 305 |

Desfecho RAG por coorte (braço bm25):

| Coorte | n | KC | Correção | Cob. de passagem |
|---|---|---|---|---|
| estável aprovado | 1383 | 0,603 | 0,645 | 0,767 |
| novo aprovado | 229 | 0,605 | 0,636 | 0,745 |
| novo reprovado | 196 | 0,588 | 0,623 | 0,746 |
| estável reprovado | 862 | 0,640 | 0,681 | 0,681 |

Procedência das passagens (bm25, único braço cujo `chunk_id` compartilha o
espaço de identificadores do CEP):

| Coorte | chunk fonte no top-1 | chunk fonte no top-k |
|---|---|---|
| estável aprovado | 44,1% | 75,3% |
| novo aprovado | 46,7% | 81,2% |
| novo reprovado | 56,1% | 84,7% |
| estável reprovado | 50,1% | 82,3% |

**Leitura.** O fix do contexto consertou a cauda de falhas do juiz: erros caíram
de 12 para 1, notas 0,0 em fidelidade de 47 para 18, e o payload por critério
encolheu 89%. Mas o efeito líquido sobre o portão é indistinguível de ruído de
re-julgamento: kappa 0,669, Δ médio nulo em todos os critérios enquanto o \|Δ\|
é grande, flips simétricos entre as duas direções (fidelidade decide 83 novas
aprovações e 54 novas reprovações), e coortes de flip sem assinatura própria no
desfecho RAG. A hipótese que motivou o fix (pares aprovados com evidência de
outro trecho da entrevista) também não aparece: nos novos reprovados o chunk
fonte é recuperado mais, não menos.

Isso não invalida o fix, que está certo por princípio e é o que a metodologia
descreve. Significa que **o portão não ficou mais seletivo para conhecimento
tácito**, e que a confiabilidade do juiz, não o contexto, é o fator dominante.

*Como reproduzir:*

```bash
uv run python -m scripts.analyze_cep_gate_flip --old thesis-run-01 --new thesis-run-02
```

**Limitação.** A comparação confunde a mudança de contexto com o
não-determinismo do juiz (`qwen3:14b`, temperatura 0,7). Para separar as duas é
preciso um teste-reteste: replicar o run e re-julgar uma amostra com a
configuração idêntica à do run-02.

```bash
uv run arandu replicate thesis-run-02 --id teste-reteste
uv run arandu judge-qa results/teste-reteste/cep/outputs --files 50 --pairs 6 --rejudge
uv run python -m scripts.analyze_cep_gate_flip --old thesis-run-02 --new teste-reteste
```

Se a taxa de troca sob repetição pura ficar perto dos 16% observados, todo o
efeito medido na seção 5 é ruído.

## 6. Ressalvas de leitura

### 6.1 Resolvida: validade e procedência dos estágios êmicos

Fica o registro, porque a ressalva valeu para textos escritos entre 2026-09-21 e
2026-09-24.

Até o run êmico de 2026-08-19, `emic_judge`, `human_eval` e `annotation`
estavam ancorados no portão CEP antigo (1579 aprovados, população por célula
1225/115/146/93) e cegos aos metadados da fonte. O run de 2026-09-23 passou a
tomar `results/thesis-run-02/cep/outputs` como entrada, e as amostras humanas
foram re-sorteadas sobre o mesmo portão, com o bloco de metadados no instrumento
(seções 4.7 e 4.8). A divergência de procedência que sobrou, entre o tarball do
Drive e a árvore local, terminou com a republicação da seção 1.1, conferida por
md5.

*Como confirmar em que estado está uma árvore:*

```bash
uv run python - <<'EOF'
import json
m = json.load(open("results/thesis-run-02/human_eval/outputs/sample_manifest.json"))
item = json.loads(open("results/thesis-run-02/human_eval/outputs/sample.jsonl").readline())
print(m["pipeline_id"], m["population_by_cell"], "metadata" in item)
# instrumento vigente: thesis-run-02 {'remember': 1263, ...} True
# instrumento cego:    thesis-run-01 {'remember': 1225, ...} False
EOF
```

### 6.2 O benchmark RAG cobre os 2670 pares, aprovados ou não

O `retrieve` deste run foi executado em 2026-07-02, antes do filtro que descarta
pares reprovados (`43124d50`, #154, de 2026-07-07). Isso precisa constar da
metodologia por dois motivos: é o que permite as coortes da seção 4.6 e da seção
5, e uma re-execução hoje produziria um benchmark restrito aos 1612 aprovados,
sem essas coortes.

### 6.3 Só o estágio `cep` foi re-julgado

`retrieve`, `answers` e `judge_answers` são byte a byte idênticos aos do run
anterior. As tabelas da seção 4.5 mudam apenas pelo refiltro do portão, e o juiz
de respostas nunca foi afetado pelo #168: ele se fundamenta em
`QAPairCEP.context` (o chunk) mais as passagens recuperadas, nunca na
transcrição inteira.

### 6.4 Um par sem nota no juiz CEP

Um par segue sem nota por falha do juiz CEP (eram 12 no run anterior). Ele
aparece como `sem avaliação` na decomposição da seção 4.2 e é excluído de toda
média. O juiz êmico, por outro lado, pontuou os 2670 pares sem nenhum erro.

## 7. Reconstrução do zero

Sequência mínima para reproduzir este documento a partir de uma máquina limpa:

```bash
git clone <repo> && cd arandu
uv sync --all-extras --all-groups

curl -L -o ~/Downloads/thesis-run-02.tgz \
  "https://drive.usercontent.google.com/download?id=1qJUuzIUYbvPbreCm-mfNEDz1ra8BK_hC&export=download&confirm=t"
mkdir -p results/thesis-run-02
tar -xzf ~/Downloads/thesis-run-02.tgz -C results/thesis-run-02
uv run arandu rebuild-index

# secao 4.1 a 4.3
cd notebooks && uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 bloom-thesis-run-02.ipynb && cd ..

# secao 4.4
uv run python -m scripts.analyze_judge_thresholds --id thesis-run-02
for style in bars density hist; do
  uv run python -m scripts.plot_judge_score_distributions --id thesis-run-02 --style $style
done

# secao 4.5
uv run arandu rag-analysis --id thesis-run-02

# secao 4.6
uv run python -m scripts.analyze_qa_cohort_rag_outcome --id thesis-run-02

# secao 4.7
uv run python scripts/analyze_emic_scores.py --id thesis-run-02 \
  --out results/thesis-run-02/analysis/outputs/emic_scores.md

# secao 4.7.1 (precisa da linha de base cega, receita na propria secao)
uv run python -m scripts.analyze_emic_metadata_effect \
  --id thesis-run-02 --baseline ~/Downloads/thesis-run-02-emic-blind

# secao 5 (precisa tambem do thesis-run-01 extraido em results/)
uv run python -m scripts.analyze_cep_gate_flip --old thesis-run-01 --new thesis-run-02
```

Nenhum desses comandos chama LLM: todos são releitura e agregação de artefatos
já persistidos. O único passo que exige GPU e horas de compute é re-executar
`arandu retrieve`, que este documento não requer.

## 8. Registro de execução

As seções 4.1 a 4.6 e a seção 5 foram executadas em 2026-09-21 sobre o arquivo
de 2026-09-21, no commit `4708d9e7`, e reconferidas em 2026-09-24 sobre o
arquivo vigente: os estágios que as sustentam são byte a byte idênticos entre as
duas versões, e os comandos reproduzem os mesmos números.

As seções 4.7, 4.7.1 e 4.8 foram executadas em 2026-09-24 sobre o arquivo
vigente, no commit `764eac15`.

Os tarballs publicados no Drive foram regerados da árvore local e republicados
em 2026-09-24, e conferidos por md5 depois do upload (seção 1.1).

Dois scripts foram acrescentados ao repositório para que as análises sejam
refazíveis: `scripts/analyze_cep_gate_flip.py` (seção 5, em 2026-09-21) e
`scripts/analyze_emic_metadata_effect.py` (seção 4.7.1, em 2026-09-24). Ambos
passam `ruff check` e `ruff format`.

Uma armadilha encontrada no caminho, que vale para quem for revisar: o `diff`
desta máquina passa por um proxy (`rtk`) que reportou "Files are identical" para
dois arquivos com md5 diferente. Toda comparação de arquivo deste documento foi
refeita com `md5sum` e `difflib`; não confie no `diff` aqui.
