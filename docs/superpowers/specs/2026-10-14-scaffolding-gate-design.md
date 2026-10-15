# Portão de admissão no scaffolding de Bloom: desenho

Data: 2026-10-14
Branch: `feature/scaffolding-gate`
Módulos afetados: `qa/cep/bloom_scaffolding.py`, `qa/cep/generator.py`, `qa/batch.py`,
`qa/config.py`, `qa/schemas.py`, `cli/qa.py`, `docker-compose.yml`

## 1. Propósito

Impedir que pares ruins condicionem a geração dos níveis seguintes da escada de
Bloom. Cada par gerado passa pelo `QAJudge` antes de poder entrar no contexto de
scaffolding; só os aprovados entram. O veredito fica gravado no par, de modo que
o `judge-qa` posterior não precisa julgá-lo de novo.

## 2. Problema

`BloomScaffoldingGenerator` alimenta o prompt de cada par com todos os pares
gerados antes no mesmo chunk:

- `bloom_scaffolding.py:166`: `prior = pairs`, entre níveis.
- `bloom_scaffolding.py:222`: `current_prior = all_prior_pairs + pairs`, entre
  irmãos do mesmo nível.

O único filtro é sintático (parse JSON e validação Pydantic em `_parse_response`).
O portão semântico só roda em `arandu judge-qa`, depois que o dataset inteiro já
foi gerado. Um `remember` alucinado ou mal calibrado entra no prompt de
`understand`, `analyze` e `evaluate` do mesmo chunk, e esse viés fica registrado
em `generation_prompt`. Como os 3 `remember` são a base da escada (3/1/1/1), o
efeito se propaga para metade do dataset cognitivo.

## 3. Decisões travadas

| Decisão | Racional |
| --- | --- |
| O veredito inline é persistido em `QAPairCEP.validation` | Com T = 0,1 o juiz não é determinístico. Descartar o veredito e julgar de novo no `judge-qa` permitiria um par admitido no scaffolding e reprovado no dataset (ou o inverso). Persistir também evita pagar o juiz duas vezes: `judge-qa --resume` já pula pares com veredito limpo. |
| `enable_scaffolding_gate` tem default `True` | O comportamento atual é o defeito. Quem precisar reproduzir o regime antigo desliga explicitamente. |
| Pares reprovados ficam no output, fora só do scaffolding | Mantém a distribuição 3/1/1/1 da geração e o denominador do `validation_rate` usado pelas análises do portão. |
| Sem regeneração de pares reprovados | Regenerar até passar é amostragem por rejeição contra o mesmo juiz que avalia o dataset. Fica fora de escopo (YAGNI). |
| Veredito ausente ou com erro de critério não admite o par | Na dúvida, o par fica de fora do scaffolding. O `judge-qa --resume` já re-julga vereditos com erro. |
| O contexto do juiz inline é construído por `build_pair_judge_context` | É a mesma função que `judge-qa` usa (`cli/qa.py:473`). Os dois portões veem o mesmo chunk e o mesmo bloco de metadados. |

## 4. Desenho

### 4.1 Contrato do validador

Em `qa/cep/bloom_scaffolding.py`, um `Protocol` estreito. `QAJudge` já o
satisfaz sem alteração:

```python
class PairValidator(Protocol):
    """Judge a generated pair against its grounding context."""

    def validate(self, qa_pair: QAPairCEP, context: str) -> QAPairCEP: ...
```

`BloomScaffoldingGenerator.__init__` ganha `validator: PairValidator | None = None`.
`CEPQAGenerator.__init__` recebe o mesmo parâmetro e só o repassa. Nenhum dos dois
importa `QAJudge`, `JudgeConfig` ou cliente de validador.

### 4.2 Fluxo por chunk

1. `generate(context, source_metadata=...)` calcula o contexto do juiz uma vez:
   `build_pair_judge_context(context, context, source_metadata,
   enable_metadata=cep_config.enable_source_metadata_context,
   language=cep_config.language)`. Só quando há validador.
2. `_generate_for_level` recebe esse `judge_context` como keyword.
3. Depois de `_generate_single_pair`, se há validador:
   `pair = self.validator.validate(pair.model_copy(update={"validation": None}),
   judge_context)`. O par julgado é o que entra em `pairs`.
4. A seleção de scaffolding passa por um método único:

   ```python
   def _scaffolding_pool(self, pairs: list[QAPairCEP]) -> list[QAPairCEP]:
       if self.validator is None:
           return pairs
       return [p for p in pairs if p.is_valid is True]
   ```

   Aplicado nos dois pontos da seção 2. O corte `pairs[-max_scaffolding_pairs:]`
   de `_format_prior_pairs` continua lá e roda depois do filtro, então um par
   reprovado não ocupa vaga.
5. O retorno de `generate()` continua contendo todos os pares, aprovados ou não.

`is_valid is True` já exclui vereditos com erro de critério: `CriterionScore.passed`
é `False` com erro, e o estágio `cep_validation` roda em modo `filter`, então
`JudgePipelineResult.passed` sai `False`. Não é preciso importar `has_judge_error`
na geração.

O enricher do Module II (`ReasoningEnricher.enrich`, `reasoning.py:130`) hoje
reconstrói o par com `QAPairCEP(...)` campo a campo e perde `validation`,
`rationale`, `generation_thinking` e `context` derivados. Com
`enable_reasoning_traces=True`, o veredito do portão sumiria antes de ser gravado.
O enricher passa a usar `qa_pair.model_copy(update=...)`.

O parse da resposta de geração (`_parse_response`) passa a forçar
`validation=None`, assim como já força `bloom_level` e `context`. Sem isso, um
LLM que ecoasse uma chave `validation` no JSON produziria um veredito falso que
o `judge-qa --resume` respeitaria.

Quando nenhum par de um nível é admitido, os níveis seguintes rodam sem
scaffolding, como já acontece com o primeiro `remember`. Um `logger.warning`
registra o nível e o tamanho do pool vazio.

### 4.3 Quando o validador é construído

O validador só existe com `enable_scaffolding_context=True` e
`enable_scaffolding_gate=True`. Sem scaffolding o portão não tem efeito, e julgar
na geração seria só antecipar o `judge-qa`.

### 4.4 Construção nos workers

`batch._init_cep_worker` e o caminho sequencial em
`generate_cep_qa_for_transcription` hoje duplicam a montagem do
`CEPQAGenerator`. Os dois passam a chamar uma função única:

```python
def _build_cep_generator(
    qa_config: QAConfig, cep_config: CEPConfig, judge_config: JudgeConfig
) -> CEPQAGenerator: ...
```

Ela monta o `LLMClient` de geração como hoje e, quando o portão está ativo,
`build_validator_client(judge_config.validator_model, provider=...,
base_url=...)` mais `QAJudge(validator_client, cep_config, judge_config)`.
`run_batch_cep_generation` e as funções de worker recebem `judge_config_dict` ao
lado de `qa_config_dict` e `cep_config_dict`, para que overrides de CLI cheguem
aos processos filhos.

### 4.5 Configuração

`CEPConfig`:

```python
enable_scaffolding_gate: bool = Field(
    default=True,
    description="Judge each generated pair with QAJudge before it can enter the "
    "scaffolding context of later pairs. Rejected pairs stay in the output with "
    "their verdict but are never shown to the generator. Requires "
    "ARANDU_JUDGE_VALIDATOR_MODEL. No effect without enable_scaffolding_context.",
)
```

Env: `ARANDU_CEP_ENABLE_SCAFFOLDING_GATE`. O validador continua configurado por
`JudgeConfig` (`ARANDU_JUDGE_VALIDATOR_MODEL`, `_PROVIDER`, `_BASE_URL`,
`_TEMPERATURE`, `_LANGUAGE`); nada é duplicado em `CEPConfig`.

### 4.6 CLI

`arandu generate-cep-qa` ganha:

- `--scaffolding-gate/--no-scaffolding-gate`, override de `enable_scaffolding_gate`.
- `--validator-model`, `--validator-provider`, `--validator-base-url`, overrides
  de `JudgeConfig` pela mesma cascata `model_validate({**dump(), **overrides})`.

Validação no início do comando: com portão e scaffolding ativos e
`validator_model` vazio, `print_error` e `typer.Exit(code=1)` antes de qualquer
chamada de LLM. A regra mora em uma função única,
`require_gate_validator(cep_config, judge_config)` em `qa/config.py`, chamada
pela CLI e por `_build_cep_generator` (que protege os workers). O bloco "CEP QA Generation Configuration" passa a mostrar o
estado do portão e o modelo validador.

### 4.7 Proveniência

- `QARecordCEP.scaffolding_gate_enabled: bool = False`, no padrão de
  `source_metadata_context_enabled`. O default `False` faz registros antigos
  carregarem com o valor correto; registros novos gravam o estado efetivo do
  portão (ativo só com scaffolding e gate ligados).
- Com o portão ativo, `QARecordCEP.validator_model_id` e `validated_pairs` são
  preenchidos na geração, do mesmo jeito que o `judge-qa` faz hoje.
- No `run_metadata.json` da etapa `cep`, `run_batch_cep_generation` acrescenta
  `config_values["scaffolding_gate"]`, ao lado de `cep_config`. O conteúdo
  reaproveita `judge_run`: `judge` (`JudgeConfig` resolvido, que inclui
  `validator_model`, `validator_provider` e `validator_base_url`), `criteria`
  via `snapshot_criteria` e `bloom_descriptions_sha256` via
  `snapshot_bloom_descriptions`. Com o portão
  desligado, a chave registra `{"enabled": false}`.

### 4.8 Interação com `judge-qa`

Nenhuma mudança de código. Em modo `--resume`, pares com veredito limpo da
geração são contados como já julgados, e pares com erro de critério são
re-julgados. `--rejudge` continua forçando um passe novo e é o caminho do
estudo de confiabilidade intra-juiz. O `judge_qa/run_metadata.json` de um
resume sobre um run com portão registra só os pares que ele mesmo julgou; os
demais estão cobertos pelo snapshot da seção 4.7.

### 4.9 Deploy

No cluster, `arandu_run_worker` (`scripts/slurm/container_lib.sh`) já repassa ao
container todas as variáveis `ARANDU_*` exportadas. Basta `cep_common.sh` exportar
`ARANDU_JUDGE_*` (mesmos defaults de `judge_common.sh`) e
`ARANDU_CEP_ENABLE_SCAFFOLDING_GATE`. Quando o validador usa Ollama e o modelo
difere de `ARANDU_QA_MODEL_ID`, o script também precisa fazer `ollama pull` do
modelo validador no sidecar, que hoje só baixa o modelo de geração.

No `docker-compose.yml`, o serviço `arandu-cep` (perfis `cep` e `cep-gpu`) passa
a declarar as mesmas variáveis no `environment:`, como o `arandu-judge` faz em
`docker-compose.yml:306`. O mapa de camadas em `scripts/slurm/AGENTS.md` deixa
de dizer que `generate-cep-qa` não valida.

## 5. Custo

Por chunk com 3/1/1/1: 6 chamadas de geração e 18 de juiz (3 `remember` com 2
critérios e 3 pares com 4). Como os vereditos são persistidos, essas 18
chamadas substituem as do `judge-qa`. O custo total do pipeline não muda; ele
passa da etapa `judge_qa` para a etapa `cep`, que fica mais longa no SLURM e
precisa de tempo de alocação ajustado.

## 6. Impacto metodológico

Datasets gerados com o portão não são comparáveis ao thesis-run-02, porque o
prompt de todo par acima do primeiro `remember` muda. O próximo run usa um id
novo. A comparação entre regimes (taxa de aprovação por nível com e sem portão)
fica possível porque `scaffolding_gate_enabled` está em cada registro.

## 7. Testes

Em `tests/qa/cep/test_bloom_scaffolding.py`, com um `PairValidator` fake
(aprovação decidida por pergunta) e `LLMClient` mockado:

- par reprovado não aparece no prompt do nível seguinte e aparece no output;
- par reprovado não aparece no prompt do irmão seguinte do mesmo nível;
- veredito `None` e veredito com `error` em critério não são admitidos;
- `max_scaffolding_pairs` é aplicado depois do filtro;
- o validador recebe o contexto igual a `build_pair_judge_context` para o chunk;
- o par é julgado sem veredito prévio (`validation` zerado antes de `validate`);
- `validator=None` produz prompts idênticos aos de hoje (regressão);
- pool vazio gera warning e prompt sem seção de scaffolding.

Em `tests/qa/`:

- `CEPConfig` aceita `enable_scaffolding_gate` e o default é `True`;
- `_build_cep_generator` constrói validador só com gate e scaffolding ativos;
- `QARecordCEP` antigo, sem o campo, carrega com `scaffolding_gate_enabled=False`;
- `run_metadata` da etapa `cep` contém `scaffolding_gate` com os digests.

Em `tests/cli/`:

- `generate-cep-qa` sai com código 1 quando o portão está ativo e não há
  modelo validador;
- `--no-scaffolding-gate` dispensa o modelo validador.

## 8. Documentação

Atualizar `src/arandu/qa/AGENTS.md` (a geração deixa de ser só geração quando o
portão está ativo), `docs/user-guide/cep-qa-generation.md`,
`docs/user-guide/cli-reference.md` e `scripts/slurm/AGENTS.md`.

## 9. Fora de escopo

- Regenerar pares reprovados para completar a cota do nível.
- Julgar com um juiz diferente do `QAJudge` do portão.
- Re-gerar o thesis-run-02.
