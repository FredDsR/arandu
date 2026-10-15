# Arandu Documentation

Welcome to the Arandu documentation. This guide will help you find the information you need based on your role.

## Quick Navigation

### New Users

Start here if you're new to Arandu:

1. **[Getting Started](guide/getting-started.md)** - Installation and first steps
2. **[Transcription Guide](guide/pipelines/transcription.md)** - Process audio/video files
3. **[Configuration Reference](guide/configuration.md)** - Environment variables and settings

### Pipeline Users

Guides for each processing pipeline:

| Pipeline | Guide | Description |
|----------|-------|-------------|
| Transcription | [transcription.md](guide/pipelines/transcription.md) | Audio/video to text |
| CEP QA Generation | [cep-qa-generation.md](guide/pipelines/cep-qa-generation.md) | Cognitive scaffolding with Bloom's Taxonomy |
| KG Construction | [kg-construction.md](guide/pipelines/kg-construction.md) | Build knowledge graphs |
| RAG Analysis | [rag-analysis.md](guide/pipelines/rag-analysis.md) | Cross-arm retrieval evaluation (Phase C) |

### Operators & DevOps

Deployment guides for different environments:

- **[Docker Deployment](operations/docker.md)** - Local Docker setup (GPU/CPU/ROCm)
- **[PCAD Cluster](operations/pcad.md)** - PCAD HPC cluster deployment
- **[SLURM Guide](operations/slurm.md)** - Generic SLURM job configuration

### Developers

Technical documentation for contributors:

- **[Architecture](development/architecture.md)** - System design and patterns
- **[Data Schemas](development/schemas.md)** - Pydantic models and validation
- **[Testing Guide](development/testing.md)** - Test suite and coverage
- **[Dependencies](development/dependencies.md)** - Package requirements
- **[CI/CD Setup](development/ci-cd.md)** - GitHub Actions workflow

### Roadmap

- **[ROADMAP.md](planning/ROADMAP.md)** — master's roadmap, phase timeline, and active tasks

## Documentation Structure

```
docs/
├── README.md                 # This file (documentation index)
├── user-guide/               # End-user documentation
│   ├── getting-started.md
│   ├── transcription.md
│   ├── cep-qa-generation.md  # CEP cognitive scaffolding pipeline
│   ├── kg-construction.md
│   ├── rag-analysis.md       # Cross-arm retrieval evaluation (Phase C)
│   ├── configuration.md
│   └── cli-reference.md
├── deployment/               # Deployment guides
│   ├── docker.md
│   ├── pcad.md
│   └── slurm.md
├── development/              # Developer documentation
│   ├── architecture.md
│   ├── schemas.md
│   ├── testing.md
│   ├── dependencies.md
│   └── ci-cd.md
└── planning/
    └── ROADMAP.md            # Master's roadmap + phase status
```

## External Resources

- [AutoSchemaKG Documentation](https://hkust-knowcomp.github.io/AutoSchemaKG/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Ollama Documentation](https://github.com/ollama/ollama)

## Contributing

See [AGENTS.md](../AGENTS.md) for development guidelines, coding standards, and the contribution workflow.
