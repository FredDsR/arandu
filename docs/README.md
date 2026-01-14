# G-Transcriber Documentation

Complete documentation for the Knowledge Graph Construction Pipeline (v2.0)

## Overview

G-Transcriber is an automated transcription and knowledge extraction system that processes audio/video files, generates synthetic QA datasets, constructs knowledge graphs, and evaluates knowledge elicitation quality.

## Documentation Structure

### Main Documentation

- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Complete implementation plan for P2 and P3 tasks
  - Executive summary
  - Architecture overview
  - Task breakdown
  - Implementation phases
  - Success criteria

### Implementation Details

Located in `docs/implementation/`:

1. **[DATA_SCHEMAS.md](implementation/DATA_SCHEMAS.md)** - Complete data schema specifications
   - QA generation schemas (QAPair, QARecord)
   - Knowledge graph schemas (KGNode, KGEdge, KGRecord)
   - Evaluation schemas (EvaluationReport)
   - Schema relationships and examples

2. **[CONFIGURATION.md](implementation/CONFIGURATION.md)** - Configuration reference
   - All configuration settings
   - Environment variables
   - Configuration examples
   - Best practices

3. **[CLI_REFERENCE.md](implementation/CLI_REFERENCE.md)** - CLI commands reference
   - Command overview
   - Detailed command documentation
   - Usage examples
   - Tips and best practices

4. **[FILE_STRUCTURE.md](implementation/FILE_STRUCTURE.md)** - Project file structure
   - Directory structure
   - Existing files to modify
   - New files to create
   - Generated files and directories

5. **[DEPENDENCIES.md](implementation/DEPENDENCIES.md)** - Dependency documentation
   - Complete dependency list
   - Installation instructions
   - Version compatibility
   - License information

## Quick Start

### For Developers

**First Time Setup**:
```bash
# Clone repository
git clone <repo-url>
cd etno-kgc-preprocessing

# Install dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your settings

# Run tests
pytest
```

**Read These First**:
1. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - Understand the overall plan
2. [FILE_STRUCTURE.md](implementation/FILE_STRUCTURE.md) - Know what to build
3. [DATA_SCHEMAS.md](implementation/DATA_SCHEMAS.md) - Understand data models

### For Users

**Installation**:
```bash
pip install -e .
```

**Basic Usage**:
```bash
# Generate QA dataset
gtranscriber generate-qa results/ -o qa_dataset/

# Build knowledge graphs
gtranscriber build-kg results/ -o knowledge_graphs/

# Evaluate quality
gtranscriber evaluate qa_dataset/ results/
```

**Read These First**:
1. [CLI_REFERENCE.md](implementation/CLI_REFERENCE.md) - Learn the commands
2. [CONFIGURATION.md](implementation/CONFIGURATION.md) - Configure the system

## Features

### Existing (v1.0)

- **Automated Transcription** - Whisper ASR models from Hugging Face
- **Google Drive Integration** - Download, transcribe, upload workflow
- **Batch Processing** - Parallel workers with checkpointing
- **Multi-Environment** - Docker support (GPU, CPU, ROCm)
- **SLURM Integration** - Optimized for HPC clusters

### New (v2.0)

- **Synthetic QA Generation** - Generate question-answer pairs from transcriptions
- **Knowledge Graph Construction** - Build graphs using AutoSchemaKG framework
- **Knowledge Evaluation** - Measure quality across four dimensions
- **Hybrid LLM Support** - OpenAI, Claude, and Ollama integration

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    G-Transcriber v2.0                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Transcription│→ │      QA      │→ │      KG      │    │
│  │   Pipeline   │  │  Generation  │  │ Construction │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│         │                 │                   │            │
│         └─────────────────┴───────────────────┘            │
│                           ↓                                │
│                  ┌──────────────┐                         │
│                  │  Evaluation  │                         │
│                  └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Status

| Phase | Status | Description |
|-------|--------|-------------|
| **Planning** | ✅ Complete | Documentation and architecture |
| **Phase 1** | 🔲 Pending | Foundation (LLM client, config, schemas) |
| **Phase 2** | 🔲 Pending | QA Generation |
| **Phase 3** | 🔲 Pending | KG Construction |
| **Phase 4** | 🔲 Pending | Evaluation |
| **Phase 5** | 🔲 Pending | Research (GraphRAG, frameworks) |

## Key Technologies

- **Language**: Python 3.13+
- **CLI Framework**: Typer
- **Configuration**: Pydantic Settings
- **LLM APIs**: OpenAI, Anthropic, Ollama
- **KG Framework**: AutoSchemaKG (atlas-rag)
- **Graph Library**: NetworkX
- **Containerization**: Docker
- **Cluster**: SLURM

## Project Structure

```
etno-kgc-preprocessing/
├── src/gtranscriber/           # Main package
│   ├── main.py                 # CLI entrypoint
│   ├── config.py               # Configuration
│   ├── schemas.py              # Data models
│   └── core/                   # Core modules
├── scripts/slurm/              # SLURM job scripts
├── docs/                       # Documentation (this directory)
├── tests/                      # Test suite
├── docker-compose.yml          # Docker services
└── pyproject.toml             # Project metadata
```

## Contributing

### Development Workflow

1. **Branch Naming**: `feature/<phase>-<description>`
2. **Commit Messages**: Follow conventional commits
3. **Code Style**: Use `black` for formatting, `ruff` for linting
4. **Testing**: Write tests for all new features
5. **Documentation**: Update docs with changes

### Code Standards

- **Python Version**: >= 3.13
- **Type Hints**: Required for all functions
- **Docstrings**: Google-style docstrings
- **Testing**: Minimum 80% coverage

## Resources

### External Documentation

- [AutoSchemaKG GitHub](https://github.com/HKUST-KnowComp/AutoSchemaKG)
- [AutoSchemaKG Paper](https://arxiv.org/abs/2505.23628)
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
- [OpenAI API](https://platform.openai.com/docs/api-reference)
- [Anthropic Claude API](https://docs.anthropic.com/claude/reference/)
- [Ollama](https://github.com/ollama/ollama)

### Research Papers

- **AutoSchemaKG**: Huang et al. (2025) - Dynamic schema induction
- **GraphRAG**: Microsoft Research - Community detection for RAG
- **Whisper**: OpenAI - Robust speech recognition

## License

MIT License - See LICENSE file

## Citation

If you use this project in research, please cite:

```bibtex
@software{gtranscriber2026,
  title={G-Transcriber: Knowledge Graph Construction Pipeline},
  year={2026},
  version={2.0}
}
```

And cite AutoSchemaKG:

```bibtex
@article{huang2025autoschemakg,
  title={AutoSchemaKG: Autonomous Knowledge Graph Construction through Dynamic Schema Induction from Web-Scale Corpora},
  author={Huang, Haoyu and others},
  journal={arXiv preprint arXiv:2505.23628},
  year={2025}
}
```

## Support

- **Issues**: https://github.com/<org>/<repo>/issues
- **Discussions**: https://github.com/<org>/<repo>/discussions
- **Email**: [Project maintainer email]

---

**Documentation Version**: 1.0
**Last Updated**: 2026-01-14
**Project Version**: 2.0 (in development)
