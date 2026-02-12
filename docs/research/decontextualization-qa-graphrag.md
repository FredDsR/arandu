# Decontextualization of QA Pairs for GraphRAG Evaluation

## Motivation

The CEP (Cognitive Elicitation Pipeline) generates QA pairs from interview transcriptions with riverine communities affected by critical climate events in southern Brazil. Many generated pairs are **context-dependent** -- they reference "the text", "as mentioned", "in the context", "the interviewee", etc. While acceptable for the `remember` level (where original context is available), this is **problematic for higher levels** (`understand`, `analyze`, `evaluate`) because these pairs will be used to evaluate a **GraphRAG system** where the original context is not readily available.

## Literature Review

### 1. Decontextualization (Choi et al., 2021)

**Reference**: Choi, E., et al. "Decontextualization: Making Sentences Stand-Alone." *TACL*, 2021.

Foundational framework for making sentences understandable outside their original document context. Defines four editing operations:

1. **Pronoun-to-NP swap**: Replace pronouns with their explicit referents
2. **Discourse marker removal**: Remove connectives that presuppose prior text ("however", "additionally")
3. **Scope bridging**: Add implicit scope or domain information
4. **Information addition**: Insert minimal background information needed for comprehension

Key insight: Decontextualization should be **minimal** -- add only what is necessary for standalone comprehension.

### 2. Molecular Facts (Gunjal & Durrett, EMNLP 2024)

**Reference**: Gunjal, A., & Durrett, G. "Molecular Facts: Desiderata for Decontextualization in LLM Fact Verification." *EMNLP*, 2024.

Two-stage pipeline:
1. **Ambiguity identification**: Detect which elements in a statement require decontextualization
2. **Minimal rewriting**: Rewrite with minimal additions to achieve standalone comprehension

Introduces two desiderata:
- **Decontextuality**: The statement must be understandable without the source document
- **Minimality**: No unnecessary information should be added

Relevant finding: LLMs can effectively perform decontextualization when given explicit instructions about what constitutes context-dependence.

### 3. DnDScore (Wanner et al., 2024)

**Reference**: Wanner, L., et al. "DnDScore: Decontextualization and Decomposition for Factuality Verification." 2024.

Joint decomposition + decontextualization approach. Decomposes complex claims into atomic, decontextualized facts that can be independently verified. Demonstrates that the order of operations matters: decompose first, then decontextualize each atomic fact.

### 4. RAGAS Framework

**Reference**: Explodinggradients. "RAGAS: Evaluation framework for Retrieval Augmented Generation."

Framework for synthetic QA evolution with explicit prohibition of context-dependent phrases. Key design decisions:

- **Forbidden phrases**: Explicitly prohibits "based on the provided context", "according to the passage", etc.
- **Independence filter**: Generated questions must be answerable without seeing the source passage
- **Clear intent**: Each question must have unambiguous intent

**Critical insight for our work**: Negative constraint prompting (explicit list of forbidden phrases) is highly effective at preventing context-dependent generation. This is simpler and more reliable than post-hoc decontextualization.

### 5. ARES (Stanford)

**Reference**: Saad-Falcon, J., et al. "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems." *Stanford*, 2023.

Retrieval-based validation approach: if a question cannot retrieve its source passage from a corpus, it may be too context-dependent or too vague. Uses:

- **Prediction-Powered Inference (PPI)**: Statistical framework for evaluation with minimal human labels
- **LLM judges**: Trained on human-annotated examples for context relevance, answer faithfulness, and answer relevance

Relevant insight: The retrievability of a passage given only the question serves as a proxy for question self-containedness.

### 6. RAGEval (ACL 2025)

**Reference**: Zhu, K., et al. "RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework." *ACL*, 2025.

Schema-based generation framework that creates evaluation datasets for RAG systems. Uses structured schemas to ensure generated QA pairs have clear entity references and are not ambiguous about what they refer to.

### 7. SciQAG

**Reference**: Wan, Y., et al. "SciQAG: A Framework for Auto-Generated Science Question Answering."

Three-stage pipeline for scientific QA:
1. **Passage selection**: Identify information-rich passages
2. **Question generation**: Generate questions with explicit entity references
3. **Answer extraction**: Extract answers grounded in the passage

Relevant pattern: Questions are generated with explicit entity naming rather than pronomial references.

### 8. KG-QAGen

**Reference**: Knowledge Graph-based QA generation framework.

Multi-level QA generation framework based on knowledge graphs. Generates questions at different complexity levels by traversing graph paths. Questions naturally reference explicit entities (graph nodes) rather than text positions.

### 9. Bloom's Taxonomy QG with LLMs (2024)

**Reference**: Research on progressive prompting strategies (PS1-PS5) for Bloom's taxonomy-calibrated question generation.

Five progressive prompting strategies:
- **PS1**: Basic generation prompt
- **PS2**: Add Bloom level description
- **PS3**: Add cognitive verbs and starters
- **PS4**: Add examples per level
- **PS5**: Add scaffolding context from lower levels

Key finding: Higher prompting strategies (PS4-PS5) produce better-calibrated questions, but without explicit decontextualization constraints, they still generate context-dependent references.

### 10. GraphRAG-Bench

**Reference**: Benchmark for evaluating GraphRAG systems.

Comprehensive benchmark that evaluates GraphRAG systems on multiple dimensions. Questions in this benchmark are designed to be answerable from graph structure, requiring explicit entity references rather than document positions.

### 11. LangChain "Standalone Question" Pattern

Common pattern in RAG pipelines where conversational questions are rewritten to be standalone. Uses an LLM to reformulate questions like "What did they do next?" into "What did the researchers at MIT do after publishing the 2023 paper on climate modeling?"

This pattern demonstrates that LLMs can effectively perform question decontextualization as a rewriting task.

### 12. Negative Constraint Prompting (RAGAS Breakdown)

Analysis of the RAGAS approach reveals that the most effective technique for preventing context-dependent generation is **negative constraint prompting**: providing an explicit list of forbidden phrases and patterns.

Advantages:
- **Simple to implement**: Just add constraints to the generation prompt
- **Highly effective**: LLMs reliably avoid explicitly forbidden patterns
- **No post-processing needed**: Questions are generated correctly from the start
- **Composable**: Can be combined with positive instructions about what TO include

## Approach Adopted

Based on the literature review, we adopt a **prompt-first + validation criterion** approach:

### Why Not Post-Processing (Module 1.5)?

1. **RAGAS evidence**: Negative constraints in generation prompts are highly effective
2. **Simpler architecture**: No additional pipeline step needed
3. **Lower cost**: No extra LLM calls for rewriting
4. **Composable architecture**: If empirically insufficient, a decontextualization module can be inserted later

### Implementation

1. **Negative constraints in generation prompts**: Explicit list of forbidden phrases ("no texto", "mencionado no texto", etc.)
2. **Positive instructions for autonomy**: Instructions to name entities, locations, and techniques explicitly
3. **Updated examples**: Examples that demonstrate self-contained questions
4. **New validation criterion**: `self_containedness` score (0.0-1.0) evaluated by LLM judge
5. **Remember-level exemption**: Context-dependence is expected and acceptable for recall questions

### Expected Outcomes

- **remember** level: Unchanged (context-dependence expected)
- **understand+** levels: Questions should name entities explicitly instead of referencing "the text"
- **Validation**: Low `self_containedness` scores for context-dependent pairs, enabling filtering
- **Backward compatibility**: Existing JSON datasets default to `self_containedness=1.0`

## References

1. Choi, E., et al. (2021). "Decontextualization: Making Sentences Stand-Alone." *TACL*.
2. Gunjal, A., & Durrett, G. (2024). "Molecular Facts: Desiderata for Decontextualization in LLM Fact Verification." *EMNLP*.
3. Wanner, L., et al. (2024). "DnDScore: Decontextualization and Decomposition for Factuality Verification."
4. Explodinggradients. "RAGAS: Evaluation framework for Retrieval Augmented Generation." GitHub.
5. Saad-Falcon, J., et al. (2023). "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems." *Stanford*.
6. Zhu, K., et al. (2025). "RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework." *ACL*.
7. Wan, Y., et al. "SciQAG: A Framework for Auto-Generated Science Question Answering."
8. "Bloom's Taxonomy Question Generation with LLMs." (2024).
9. "GraphRAG-Bench: Comprehensive Benchmark for GraphRAG Systems."
10. LangChain Documentation. "Standalone Question Pattern."
