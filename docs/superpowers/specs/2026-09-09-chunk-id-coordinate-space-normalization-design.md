# Normalização do espaço de coordenadas dos offsets

**Data:** 2026-09-09
**Issue:** [#166](https://github.com/FredDsR/arandu/issues/166), achado lateral
**Depende de:** PR [#168](https://github.com/FredDsR/arandu/pull/168) (juiz por chunk), independente em código
**Run afetado:** `thesis-run-01` -> `thesis-run-02`

## 1. Objetivo

Fazer com que todo produtor e todo consumidor de offsets no pipeline leia o
mesmo texto, e migrar os artefatos do `thesis-run-01` cujos `chunk_id` foram
computados contra o texto errado.

## 2. O defeito

`chunk_id` é derivado dos offsets (`shared/chunking/chonkie_adapter.py:11`):

```python
payload = f"{source_file_id}|{chunker_id}|{start_char}|{end_char}".encode()
return hashlib.sha1(payload).hexdigest()[:16]
```

Dois lugares chunkam textos diferentes por um caractere:

| Local | Texto |
| --- | --- |
| `shared/chunking/batch.py:180` | `record.transcription_text` |
| `qa/cep/generator.py:80` | `transcription.transcription_text.strip()` |

O Whisper prefixa a transcrição com um espaço. Nos 214 arquivos do
`thesis-run-01`, o primeiro caractere é `' '` em 214/214 e não há whitespace
final em nenhum. Logo toda fronteira do `chunk` stage está deslocada em `+1`
relativa à geração CEP, e todos os `chunk_id` diferem.

Medido no `thesis-run-01`:

```
pares CEP cujo chunk_id existe no ChunkSet:            0/2670
sha256 do ChunkSet == sha(texto do enriched como está): 214/214
sha256 do ChunkSet == sha(texto stripado):               0/214
deslocando os offsets do stage em -1 e recomputando
  o sha1, ids recuperados:                          2670/2670
```

A última linha prova que não é chunking divergente: é o mesmo particionamento
com as fronteiras deslocadas.

### 2.1 Por que importa

Hoje o defeito é latente. `QAPairCEP.chunk_id` só é usado como segmento opaco
dentro de um id composto (`shared/rag/analysis/loader.py:57`,
`shared/rag/judge_answers/gold_lookup.py:74`,
`qa/non_answerable/perturbation.py:86`), e os três derivam do mesmo
`QARecordCEP`, então concordam entre si. Ninguém resolve o `chunk_id` contra o
`ChunkSet`.

Onde ele morde é na próxima métrica óbvia. O retriever BM25 indexa a partir do
`ChunkSet` (`shared/rag/retrievers/bm25.py:138`), então os
`RetrievedPassage.chunk_id` estão no namespace do stage. Medir recall de
recuperação ("o BM25 trouxe o chunk que gerou esta pergunta?") daria 0% em
todos os casos, silenciosamente, sem erro.

O campo também é documentado como "Reference into the source ChunkSet"
(`qa/schemas.py`), e não satisfaz o que promete.

## 3. Decisões

1. **Normalizar para o espaço stripado**, não para o cru. A direção decide o
   tamanho da migração: o `qa_pair_id` já vive no espaço da geração (stripado)
   e está gravado em ~15 mil nomes de arquivo. Normalizar para cru mudaria
   todos eles. Normalizar para stripado deixa o `qa_pair_id` intacto.
2. **Canônico no schema**, via `@field_validator` em
   `EnrichedRecord.transcription_text`, não num helper chamado pelos
   consumidores. São cinco consumidores do mesmo espaço de coordenadas
   (`chunking/batch.py`, `qa/cep/generator.py`, `shared/rag/answer/resolver.py`,
   `kg/passage_offsets.py`, `shared/rag/retrieve/factory.py`); um helper pode
   derivar de novo, que é exatamente como chegamos aqui.
3. **Clonar o run** em vez de reescrever no lugar. `thesis-run-01` fica
   congelado como registro do estado pré-correção.
4. **O rejudge do #168 cai no mesmo run novo.** Os vereditos divergem de fato,
   não só as chaves, então um único `thesis-run-02` carrega as duas correções.

## 4. Mudança de código

### 4.1 O validador

Em `shared/schemas.py`, no `EnrichedRecord`:

```python
@field_validator("transcription_text")
@classmethod
def _normalize_transcription_text(cls, v: str) -> str:
    """Strip surrounding whitespace to fix one canonical coordinate space."""
    return v.strip()
```

Não existe `model_construct` em nenhum ponto do `src/`, verificado, então o
validador roda em toda construção e em todo `model_validate_json`. Os 214
arquivos de `transcription/outputs/` passam a carregar canônicos **sem serem
reescritos**.

### 4.2 Efeito nos consumidores

| Arquivo | Mudança |
| --- | --- |
| `qa/cep/generator.py:80` | remove o `.strip()`, agora redundante |
| `shared/chunking/batch.py:180` | nenhuma, passa a ler o texto canônico |
| `shared/rag/answer/resolver.py:97` | nenhuma, idem |
| `kg/passage_offsets.py` | nenhuma, idem |
| `shared/rag/retrieve/factory.py:217` | nenhuma, idem |

Os quatro primeiros **gravam** offsets. O quinto,
`_build_chunk_resolver`, é o consumidor no outro sentido: monta o
`ChunkResolver` que fatia `record.transcription_text` pelos spans já
persistidos nas ChunkSets para produzir o texto que o índice BM25 tokeniza. Ele
lê o mesmo espaço de coordenadas, então entra na mesma lista.

O único outro consumidor que hasheia esse texto é `chunking/batch.py:181`, e é
o hash que queremos que mude.

### 4.3 Testes

- Validador: whitespace ao redor é removido na construção e no
  `model_validate_json`.
- **Invariante**: um `EnrichedRecord` com espaço inicial, passado pelo chunker
  do `chunk` stage e pelo `_chunk_with_offsets` do gerador CEP, produz
  `chunk_id` idênticos. Este é o teste que teria pegado o defeito.

## 5. Script de migração

`scripts/migrate_chunk_id_namespace.py`, seguindo o precedente do
`scripts/kg_relabel_predicate.py`. Interface: `--id <run>`, `--dry-run`,
`--verify`, `--allow-original`.

O precedente do `kg_relabel_predicate.py` inclui um `.bak` antes de escrever;
aqui o substituto é o clone do run, e o clone só existe se o operador se
lembrar de fazê-lo. Para não deixar isso na cabeça de ninguém, escrever exige
proveniência de clone: o `pipeline.json` do run precisa carregar
`replicated_from` (gravado pelo `arandu replicate`, `shared/results_manager.py`),
senão o script recusa com código 2 e sugere `--allow-original`. `--dry-run` e
`--verify` são somente leitura e não passam por essa exigência, justamente para
que o ensaio contra o `thesis-run-01` congelado siga possível.

`main()` sempre roda a passada inteira em modo `dry_run=True` antes da passada
que escreve. Nenhum dos passos tem atomicidade entre arquivos, então todo
aborto de lógica precisa acontecer antes do primeiro byte escrito, não no meio
dos 214 ChunkSets.

### 5.1 Abordagem

Rechunkar com o chunker real e escrever no lugar, não deslocar offsets
aritmeticamente. O artefato passa a ser exatamente o que o pipeline corrigido
produziria, em vez de um resultado de aritmética que ninguém validou. O shift
esperado entra como **asserção de segurança**: se o rechunk não reproduzir o
deslocamento previsto, o script aborta antes de escrever.

Rejeitadas: shift aritmético puro (grava artefato não verificado contra o
chunker) e re-rodar o estágio `chunk` (o `ResultsManager` cria um run novo em
vez de escrever no clone).

### 5.2 Armadilha: o script precisa dos dois textos

Depois do validador da §4.1, o texto cru **deixa de ser observável através do
schema**: carregar um `EnrichedRecord` já devolve o texto stripado. Mas o
script precisa do cru para computar o deslocamento (`lead_ws`) usado no passo
4 e na asserção de segurança do passo 1.

Logo o script lê o arquivo de transcrição de duas formas:

- via `EnrichedRecord.model_validate_json` para o **texto canônico** (rechunk,
  `source_text_sha256`);
- via `json.loads` cru para o **texto original**, de onde tira
  `lead_ws = len(raw) - len(raw.lstrip())`.

O caminho do arquivo sai de
`shared.io.resolve_transcription_path(transcription_dir, file_id)`, que já é a
fonte única da convenção de nome no lado da leitura. Trata `None` como aborto,
não como skip: um `ChunkSet` sem transcrição correspondente é inconsistência
que o script não deve encobrir.

### 5.3 Passos

1. Para cada `chunk/outputs/<view>/<file_id>.json`: resolve a transcrição,
   classifica o `source_text_sha256` gravado, roda `get_chunker(view)` sobre o
   texto canônico, grava o `ChunkSet` novo (ids, offsets e
   `source_text_sha256`), e acumula `old_id -> new_id`. Antes de escrever,
   afirma que cada fronteira nova é a antiga menos `lead_ws`; se não for,
   aborta sem tocar em disco.

   A classificação do hash gravado tem três saídas: igual ao sha do texto
   canônico significa **já migrado**, então o arquivo é pulado (e reporta lead
   0, para o passo 4 não deslocar duas vezes) e o script fica idempotente;
   igual ao sha do texto cru significa **entrada da migração**, segue;
   nenhum dos dois significa que aquele `ChunkSet` não foi construído a partir
   daquela transcrição, e aí aborta com essa mensagem em vez de com um span
   inesperado. Recuperar uma migração que falhou no meio é re-clonar, nunca
   rerodar.
2. `retrieve/indexes/bm25_*/manifest.json`: remapeia `chunk_ids` por lookup no
   mapa, não por posição. O `bm25.pkl` não é tocado e o `sha256` do manifest
   cobre o pkl, não o manifest, então segue válido.
3. `{retrieve,answers,judge_answers}/outputs/**/*.json`: remapeia
   `passages[].chunk_id` em passada única sobre os valores originais. Ids fora
   do mapa ficam intocados, o que preserva `atlas_rag` e `khop_passage`
   (`<file_id>:<index>`), `khop_triple` (`triple:<sha>`) e `null` (sem
   passages) sem codificar nome de arm.
4. `kg/outputs/passage_offsets.json`: desloca `start_char`/`end_char` pelo
   `lead_ws` do arquivo de origem, com clamp em 0. Antes de escrever, afirma
   que o span novo resolve para o mesmo texto que o antigo resolvia (§5.5,
   item 5). Um `source_file_id` ausente do mapa `file_id -> lead_ws` é
   **aborto**, não skip: o mapa vem do estágio `chunk` e o sidecar vem do
   estágio `kg`, que selecionam arquivos de entrada de forma independente
   (`shared/chunking/batch.py` glob `*.json`, `kg/batch.py` glob
   `*_transcription.json`), e uma chave faltando é indistinguível de um
   `lead_ws == 0` legítimo. Um `lead_ws == 0` de verdade segue sendo no-op.

### 5.4 Alcance medido

| Artefato | Arquivos | O que muda |
| --- | --- | --- |
| `chunk/outputs/cep_4k/` | 214 | `chunk_id`, offsets, `source_text_sha256` |
| `retrieve/indexes/bm25_cep_4k/manifest.json` | 1 | 445 `chunk_ids` |
| `{retrieve,answers,judge_answers}/outputs/bm25/` | 9012 | só `passages[].chunk_id` |
| `kg/outputs/passage_offsets.json` | 1 | 302 offsets |

Intocados: `cep/outputs/`, `non_answerable/`, os arms `atlas_rag`,
`khop_passage`, `khop_triple` e `null`, `analysis/` (só PNGs), e **todos os
nomes de arquivo**.

### 5.5 Verificação

`--verify` roda depois da migração, afirma o seguinte, e falha se qualquer uma
não valer:

1. **Todo** par CEP tem `chunk_id` presente no `ChunkSet` do seu arquivo. No
   `thesis-run-02` isso significa 2670/2670, contra 0/2670 antes. O script
   reporta a fração (`verify` devolve `resolved` e `total`, e `main()` imprime
   `resolved/total` em toda execução de `--verify`) e exige que seja total, sem
   número fixo no código. `total == 0` é falha: sem par nenhum para resolver, a
   asserção principal não rodou, e um `--verify` verde não significaria nada.
2. `source_text_sha256` de cada `ChunkSet` bate com o sha do texto canônico.
3. Re-chunkar o texto canônico reproduz exatamente os ids em disco, ou seja o
   artefato é o que o pipeline corrigido produziria. A preservação de conteúdo
   não é checada aqui: `--verify` roda depois da reescrita e não vê o estado
   anterior. Ela é afirmada durante a migração, por `_assert_pure_shift`, onde
   as duas chunkagens estão em mão. Spans idênticos sobre o mesmo texto
   resolvem para as mesmas strings, com a única exceção do primeiro chunk, que
   perde o whitespace inicial. É isso que garante que scores e ranking do BM25
   não mudam: o chunk 1 antigo era `raw[3837:7535]` e o novo é
   `stripped[3836:7534]`, o mesmo texto, e BM25 não tokeniza espaço.
4. Nenhuma referência pendurada: todo id derivado de offset citado por um
   manifest do BM25 ou por `passages[].chunk_id` resolve para um chunk.
5. **Texto resolvido do `passage_offsets.json` inalterado**: cada span
   deslocado resolve para a mesma string que o span antigo resolvia, com a
   única exceção do whitespace inicial que o schema removeu (um span que
   entrava no lead o perde, igual ao primeiro chunk). Ao contrário das
   ChunkSets, aqui o estado anterior **está em mão** na hora da migração: o
   script já lê as duas leituras da transcrição, então o span antigo continua
   resolvível. A checagem vive portanto em `_assert_offset_preserved`, chamada
   por `shift_passage_offsets` antes de escrever, no mesmo lugar em que
   `_assert_pure_shift` faz o seu trabalho. O que `--verify` consegue afirmar
   depois é só o mais fraco, e afirma: todo offset cai dentro do seu texto
   canônico.

## 6. Execução

Depois de mergeados o #168 e este PR:

```bash
arandu replicate thesis-run-01 --id thesis-run-02
uv run python scripts/migrate_chunk_id_namespace.py --id thesis-run-02 --dry-run
uv run python scripts/migrate_chunk_id_namespace.py --id thesis-run-02
uv run python scripts/migrate_chunk_id_namespace.py --id thesis-run-02 --verify
arandu judge-qa results/thesis-run-02/cep/outputs --rejudge   # cluster, custa LLM
```

`arandu replicate` (`cli/manage.py:117`) já copia a árvore, reescreve
`pipeline_id`/`run_id` em cada `run_metadata.json` e grava `replicated_from` no
`pipeline.json` do clone como proveniência. O clone nasce internamente
consistente, sem `cp -r`. Essa proveniência é o que o script exige para
escrever, então a ordem acima não é só convenção: sem o `replicate` na frente,
o script recusa.

**A migração precisa vir antes de qualquer re-save de transcrição.** O texto
cru é a única fonte do `lead_ws` (§5.2), e ele só sobrevive porque
`transcription/outputs/` não é reescrito (§9). Mas todo caminho que carrega um
`EnrichedRecord` e o grava de volta agora persiste o texto **stripado**: é o
caso do `arandu judge-transcriptions` (`cli/transcribe.py`, ~linha 720, que faz
`EnrichedRecord(**data)` e depois `model_dump_json`). Rodar
`judge-transcriptions --rejudge` no clone antes da migração torna a migração
inexecutável, porque `lead_ws` passa a ser 0 para os arquivos regravados. Falha
segura, e desde a passada de validação do §5 o aborto vem antes de qualquer
escrita: o `source_text_sha256` gravado deixa de bater com qualquer das duas
leituras e o `_classify_chunk_set` do passo 1 aborta dizendo exatamente isso.
Mesmo assim, uma ocorrência parcial deixa o run sem migração possível e a
recuperação é re-clonar. Então: migrar primeiro, re-julgar depois. Não reescrever `transcription/outputs/` continua desnecessário; agora é
também inevitável em qualquer re-save.

## 7. Docs

- README, seção `Run Results`: passa a referenciar `thesis-run-02` na tabela de
  tarballs e nos exemplos de comando. Só a referência nova, sem parágrafo
  explicando a migração.
- `docs/development/architecture.md`: registra a invariante do espaço de
  coordenadas canônico, como documentação de arquitetura.

## 8. Dependência externa

Gerar e publicar `thesis-run-02.tgz` e `thesis-run-02_with-indexes.tgz` no
Drive é passo manual do Fred, posterior a este PR. Existe portanto uma janela
em que o README nomeia tarballs que ainda não estão publicados. Aceito
conscientemente; não há nada no código que resolva isso.

## 9. Fora de escopo

- Reescrever `transcription/outputs/`. O validador torna desnecessário, e o
  texto cru precisa continuar em disco para a migração ler o `lead_ws` (ver a
  ordem obrigatória na §6).
- Tocar em `QAJudge.validate_batch`, que assume um contexto compartilhado por
  vários pares. Sem chamadores em `src/`, e o #168 já documenta a restrição.
- Qualquer métrica nova de recall de recuperação. A normalização a
  desbloqueia; construí-la é outro trabalho.
