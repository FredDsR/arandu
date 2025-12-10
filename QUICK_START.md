# Quick Start: Batch Transcription

## TL;DR

Transcrever todos os arquivos de áudio e vídeo do catálogo:

```bash
gtranscriber batch-transcribe input/catalog.csv --credentials credentials.json --workers 4
```

## Pré-requisitos

1. **Instalar FFmpeg** (para extrair duração de mídia):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # macOS
   brew install ffmpeg
   ```

2. **Credenciais do Google Drive**:
   - Baixar de [Google Cloud Console](https://console.cloud.google.com/)
   - Salvar como `credentials.json` na raiz do projeto

3. **Catálogo CSV** com metadados dos arquivos (`input/catalog.csv`)

## Uso Básico

### Processamento Sequencial (1 trabalhador)
```bash
gtranscriber batch-transcribe input/catalog.csv --credentials credentials.json
```

### Processamento Paralelo (4 trabalhadores)
```bash
gtranscriber batch-transcribe input/catalog.csv --credentials credentials.json --workers 4
```

### Com Quantização (para economizar VRAM)
```bash
gtranscriber batch-transcribe input/catalog.csv --credentials credentials.json --workers 4 --quantize
```

### Modelo Mais Rápido (Turbo)
```bash
gtranscriber batch-transcribe input/catalog.csv \
  --credentials credentials.json \
  --model-id openai/whisper-large-v3-turbo \
  --workers 4
```

### Processamento em CPU (sem GPU)
```bash
gtranscriber batch-transcribe input/catalog.csv --credentials credentials.json --cpu --workers 2
```

### Diretório de Saída Personalizado
```bash
gtranscriber batch-transcribe input/catalog.csv \
  --credentials credentials.json \
  --output-dir transcricoes/ \
  --workers 4
```

## Retomar Processamento Interrompido

Se o processo for interrompido, basta executar o mesmo comando novamente. O sistema automaticamente:
- Carrega o checkpoint
- Ignora arquivos já processados
- Continua de onde parou

```bash
# Mesmo comando retoma automaticamente
gtranscriber batch-transcribe input/catalog.csv --credentials credentials.json --workers 4
```

## Recomeçar do Zero

Para processar todos os arquivos novamente:

```bash
rm results/checkpoint.json
gtranscriber batch-transcribe input/catalog.csv --credentials credentials.json --workers 4
```

## Formato de Saída

Arquivos são salvos em `results/` com o formato:
- Nome: `{gdrive_id}_transcription.json`
- Conteúdo: JSON com transcrição completa + metadados + duração

Exemplo:
```json
{
  "gdrive_id": "1JtK...",
  "name": "audio.m4a",
  "duration_milliseconds": 120000,
  "transcription_text": "Texto completo da transcrição...",
  "detected_language": "pt",
  "segments": [...]
}
```

## Monitoramento

O sistema exibe progresso em tempo real:

```
INFO - Total files: 817
INFO - Already completed: 0
INFO - Remaining to process: 817
INFO - Using 4 parallel workers
INFO - ✓ Completed: audio1.m4a
INFO - Progress: 1/817 files
INFO - ✓ Completed: audio2.m4a
INFO - Progress: 2/817 files
...
```

## Dicas de Performance

### Para GPU (Recomendado)
- Use 4-8 workers dependendo da VRAM disponível
- Use `--quantize` para economizar VRAM
- Modelo turbo para velocidade: `--model-id openai/whisper-large-v3-turbo`

### Para CPU
- Use 2-4 workers dependendo da RAM disponível
- Use modelo distilled: `--model-id distil-whisper/distil-large-v3`
- Adicione `--cpu` para forçar processamento em CPU

## Solução de Problemas

### Erro de memória
```bash
# Reduzir workers
gtranscriber batch-transcribe input/catalog.csv --credentials credentials.json --workers 2 --quantize
```

### FFmpeg não encontrado
```bash
sudo apt-get install ffmpeg
```

### Credenciais inválidas
```bash
# Verificar arquivo
ls credentials.json

# Especificar caminho
gtranscriber batch-transcribe input/catalog.csv --credentials /caminho/para/credentials.json
```

## Mais Informações

Para documentação completa, consulte:
- [BATCH_TRANSCRIPTION_GUIDE.md](BATCH_TRANSCRIPTION_GUIDE.md) - Guia completo
- [README.md](README.md) - Documentação geral do projeto
