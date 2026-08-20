# Amostra humana estratificada só por Bloom

**Data:** 2026-08-19
**Reverte:** o desenho amostral fechado em 2026-08-07 (banda êmica × Bloom)
**Registro da decisão:** `~/.cortex/workspaces/FredDsR-arandu/sessions/emic-validity-protocol/tasks/bloom-only-sampling-DECISION.md`
**Spec canônica do protocolo:** `sessions/emic-validity-protocol/spec.md` §5 (working doc, fora do repo)

## 1. Objetivo

Tirar o run do juiz êmico do caminho crítico da anotação, mudando a
estratificação da amostra humana de **banda êmica × Bloom** (8 células × 15) para
**só Bloom** (4 células × 30), e passando a montá-la a partir dos registros do
CEP em vez das saídas do `emic-judge`. O total de 120 pares é preservado: é o
compromisso assumido com os três antropólogos.

## 2. Decisões

| Decisão | Data | Motivo |
| --- | --- | --- |
| Estratificação **só por Bloom**, 4 × 30 = 120 | 2026-08-19 | A banda vinha do `emic_score`, o que colocava o run do juiz no caminho crítico. A fila do pcad materializou esse risco de cronograma. |
| Amostra montada **só dos registros do CEP** | 2026-08-19 | `QAPairCEP` já carrega `bloom_level`; `is_valid` vem do `JudgeResultMixin`, que já era lido dali como cópia autoritativa. Nada mais era necessário do juiz. |
| **`emic_score` sai do `SampleItem`** | 2026-08-19 | É o campo que devolveria a dependência pela porta dos fundos. Os scores entram na análise por junção em `pair_id`. |
| **`cell_id` sai do `SampleItem`** | 2026-08-19 | Passaria a ser byte a byte igual a `bloom_level`. Dois campos com o mesmo valor divergem depois. |
| `PER_CELL` de 10 para 30 | 2026-08-19 | O comando sem flag passa a produzir os 120 acordados. O default antigo (10 por célula, 8 células) produzia 80, não 40; os 40 só aparecem se se multiplicar as 4 células novas pelo `PER_CELL` antigo, misturando os dois desenhos. Os 120 acordados não eram alcançáveis sem flag antes de `--per-cell` existir. |
| Mudança **no lugar**, sem flag de compatibilidade | 2026-08-19 | Uma flag `--stratify bloom\|band` manteria viva no código uma decisão revertida, dobraria a matriz de testes e preservaria a dependência do juiz no caminho de código que supostamente saiu. Não existe nenhum `sample.jsonl` construído para quebrar. |
| Filtro `FRAME_BLOOM_LEVELS` **fica** | 2026-08-19 | `BloomLevel` admite seis valores; o corpus tem quatro. O filtro evita que um par `apply` abra uma quinta célula em silêncio. |
| `--per-cell` absorvido, branch antiga fechada sem PR | 2026-08-19 | O commit `9507f852` foi escrito para 8 células e a prosa que ele corrigiu fica obsoleta com 4. |

## 3. Custo aceito

A amostra passa a **espelhar a população**. Como o frame é o conjunto aprovado
pelos quatro canônicos, a população pende para o que seria a banda `limpa`, então
a amostra fica dominada por casos fáceis e a concordância alta é em parte
artefato do desenho. A ponta baixa da escala (1-2) fica com poucas observações, e
é onde o coeficiente é mais sensível. **Vai declarado na seção de limitações**,
junto do motivo.

O kappa ponderado juiz × anotador não se perde: o `emic-judge` continua
necessário e continua rodando, só deixa de alimentar a amostra. O que se perde é
a garantia de cobertura da escala.

Em troca, o desenho amostral deixa de estar acoplado ao instrumento sob teste:
trocar modelo ou prompt do juiz êmico não invalida mais a amostra.

## 4. Frame e pool

A entrada passa de dois estágios para um. `run_build_sample_batch` varre
`results/<id>/cep/outputs/*_cep_qa.json` e não abre mais
`results/<id>/emic_judge/outputs/`. Ausência do diretório do CEP vira
`FileNotFoundError` nomeando `arandu generate-cep-qa --id <run>`.

Para cada registro, itera `qa_pairs` com o índice. O par entra no pool se passar
por dois filtros, nesta ordem:

1. `pair.is_valid is not True` → descartado, soma em `excluded_not_approved`. O
   frame continua sendo o corpus aprovado pelo `judge-qa`, lido da cópia
   autoritativa no registro.
2. `pair.bloom_level not in FRAME_BLOOM_LEVELS` → descartado, soma em
   `excluded_bloom` por nível.

`pair_id` segue `f"{record.source_file_id}:{pair_index}"`, idêntico ao atual, que
é o que mantém possível a junção com as saídas do juiz. A guarda de `pair_id`
duplicado fica.

## 5. Células e seleção

`BANDS`, `DUBIOUS_MAX_SCORE` e `band_for()` saem de `sampling.py`.
`all_cell_ids()` devolve os quatro níveis de Bloom do frame, e a célula de um par
**é** o seu `bloom_level`, sem composição.

O mecanismo de seleção não muda: ordenação determinística por
`SHA-256(f"{seed}:{pair_id}")` dentro de cada célula, `per_cell` primeiros. É o
que garante reprodutibilidade a partir do seed e estabilidade entre versões de
Python. Sai a sobreamostragem 50/50, que existia só para balancear bandas.

`InsufficientCellError` continua falhando por desenho, nomeando a célula e as
contagens. O perfil de risco muda: a célula candidata a estourar deixa de ser
`duvidosa` e passa a ser o nível de Bloom mais raro. **Isso agora é verificável
antes de qualquer anotador**, porque depende só do CEP. A população de referência
do `test-kg-04` (Remember 433, Understand 644, Analyze 644, Evaluate 689) sugere
folga confortável para 30, mas com uma ressalva: o frame é o subconjunto
**aprovado pelo juiz**, não o corpus gerado, então esses números só valem como
folga se já forem posteriores ao `judge-qa`; caso contrário sobrestimam a
folga pela taxa de aprovação. Quem decide a viabilidade é o nível de Bloom
mais raro depois da aprovação, e é exatamente isso que a medição descrita a
seguir nesta seção verifica antes de convidar anotadores.

A medição sobre o `thesis-run-01` **não roda no ambiente de desenvolvimento**: os
registros do CEP desse run vivem na máquina que executa o pipeline, e é lá que
todo o fluxo de anotação roda. Então ela é o primeiro passo **da execução**, não
da implementação: sai do `build-human-eval-sample` naquela máquina, e é o gate
antes de convidar anotadores. Os testes desta mudança não dependem dela, porque
usam fixtures.

## 6. Esquema dos artefatos

`SampleItem` fica com `pair_id`, `source_file_id`, `pair_index`, `segment`,
`question`, `answer`, `bloom_level` e `slot_id` (`0..per_cell-1`). Perde
`emic_score` e `cell_id`.

`SampleManifest`:

- perde `excluded_none_score`: nenhum score é lido, então nada pode ser nulo;
- mantém `excluded_not_approved` com **significado documentado como diferente**:
  passa a contar todo par não aprovado do corpus, não só os pontuados, então a
  magnitude sobe e as contagens dos dois desenhos não são comparáveis;
- `cell_counts` e `population_by_cell` passam a ser mapas de 4 chaves, e as chaves
  **são os próprios níveis de Bloom** (`remember`, `understand`, `analyze`,
  `evaluate`), não mais `"{bloom}:{banda}"`;
- `pool_sha256` mantém a construção, mas o digest não atravessa os dois desenhos,
  porque o modelo do pool mudou;
- `input_source` no `run_metadata.json` passa a apontar para o diretório do CEP.

`HumanEvalSampleConfig` (`seed`, `per_cell`) fica como está.

## 7. CLI

`--per-cell` exposto, mínimo 1, default `PER_CELL` (30), e o comando ecoa o
tamanho resultante antes de construir.

## 8. Testes

TDD, testes antes, em `tests/shared/human_eval/`:

- **O teste que trava a decisão:** um run **sem** o diretório `emic_judge/`
  constrói a amostra normalmente. É a regressão que quebra se alguém
  reintroduzir a dependência.
- Pool montado só do CEP; par não aprovado descartado e contado; Bloom fora do
  frame descartado e contado por nível.
- `per_cell=30` → 120 itens, com balanceamento por célula e faixa de `slot_id`
  conferidos, não só o total.
- `InsufficientCellError` nomeando a célula curta.
- Mesmo seed → saída idêntica; payload alterado → `pool_sha256` diferente.
- Saem os testes de banda (`band_for`, sobreamostragem 50/50).

## 9. Fora de escopo

- **Capítulo de metodologia da dissertação** (`FredDsR-dissertacao-tex`) afirma
  estratificação por banda e precisa ser corrigido antes da entrega. Outro repo,
  outra sessão, registrado como follow-up na task de decisão.
- **CLI `emic-analysis`** (junção dos scores por `pair_id` e cálculo dos
  coeficientes) continua diferido. Esta mudança não o cria; só garante que o
  `pair_id` que ele vai precisar continua no `sample.jsonl`.
