# G-Transcriber Documentation

Welcome to the G-Transcriber documentation. This guide will help you find the information you need based on your role.

## Quick Navigation

### New Users

Start here if you're new to G-Transcriber:

1. **[Getting Started](user-guide/getting-started.md)** - Installation and first steps
2. **[Transcription Guide](user-guide/transcription.md)** - Process audio/video files
3. **[Configuration Reference](user-guide/configuration.md)** - Environment variables and settings

### Pipeline Users

Guides for each processing pipeline:

| Pipeline | Guide | Description |
|----------|-------|-------------|
| Transcription | [transcription.md](user-guide/transcription.md) | Audio/video to text |
| QA Generation | [qa-generation.md](user-guide/qa-generation.md) | Generate question-answer pairs |
| CEP QA Generation | [cep-qa-generation.md](user-guide/cep-qa-generation.md) | Cognitive scaffolding with Bloom's Taxonomy |
| KG Construction | [kg-construction.md](user-guide/kg-construction.md) | Build knowledge graphs |
| Evaluation | [evaluation.md](user-guide/evaluation.md) | Measure quality metrics |

### Operators & DevOps

Deployment guides for different environments:

- **[Docker Deployment](deployment/docker.md)** - Local Docker setup (GPU/CPU/ROCm)
- **[PCAD Cluster](deployment/pcad.md)** - PCAD HPC cluster deployment
- **[SLURM Guide](deployment/slurm.md)** - Generic SLURM job configuration

### Developers

Technical documentation for contributors:

- **[Architecture](development/architecture.md)** - System design and patterns
- **[Data Schemas](development/schemas.md)** - Pydantic models and validation
- **[Testing Guide](development/testing.md)** - Test suite and coverage
- **[Dependencies](development/dependencies.md)** - Package requirements
- **[CI/CD Setup](development/ci-cd.md)** - GitHub Actions workflow

### Implementation Planning

Active implementation guides (Phase 2 of 5 complete):

- **[Implementation Plan](planning/IMPLEMENTATION_PLAN.md)** - Master plan for all phases
- **[Phase Status](planning/PHASE_STATUS.md)** - Current progress and completed phases
- **[Quick Start for Implementers](planning/IMPLEMENTATION_QUICKSTART.md)** - Getting started with development

## Documentation Structure

```
docs/
├── README.md                 # This file (documentation index)
├── user-guide/               # End-user documentation
│   ├── getting-started.md
│   ├── transcription.md
│   ├── qa-generation.md
│   ├── cep-qa-generation.md  # CEP cognitive scaffolding pipeline
│   ├── kg-construction.md
│   ├── evaluation.md
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
└── planning/                 # Implementation planning (active)
    ├── IMPLEMENTATION_PLAN.md
    ├── PHASE_STATUS.md
    └── IMPLEMENTATION_QUICKSTART.md
```

## External Resources

- [AutoSchemaKG Documentation](https://hkust-knowcomp.github.io/AutoSchemaKG/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Ollama Documentation](https://github.com/ollama/ollama)

## Contributing

See [AGENT.md](../AGENT.md) for development guidelines, coding standards, and the contribution workflow.
