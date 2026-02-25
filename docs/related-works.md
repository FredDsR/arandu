# Related Works

This document surveys the literature relevant to the G-Transcriber pipeline and the Etno-KGC research program. The works are organized along thematic axes that correspond to the pipeline's four phases and its broader interdisciplinary context.

---

## 1. Tacit Knowledge Elicitation with LLMs

The concept of tacit knowledge -- experiential, context-dependent understanding that resists explicit articulation (Polanyi & Sen, 2009) -- is central to this research. A growing body of work investigates whether and how Large Language Models can serve as instruments for surfacing this knowledge.

**Rank et al. (2025)** present a literature review on the role of LLMs in knowledge management within manufacturing (Industry 5.0), focusing on their capacity to elicit tacit knowledge through natural language engagement with operators. They argue that traditional extraction approaches demand substantial time and personnel, positioning conversational LLM systems as a scalable alternative. **Zuin et al. (2025)** propose an agent-based framework that iteratively reconstructs dataset descriptions through interactions with employees, modeling knowledge dissemination as a Susceptible-Infectious (SI) process. Across 864 simulations, the agent achieves 94.9% full-knowledge recall, demonstrating automated tacit knowledge discovery at organizational scale.

From a philosophical perspective, **Lu (2025)** challenges the "stochastic parrot" critique by identifying two forms of tacit knowledge present in LLMs: knowledge that could theoretically be codified but is costly to translate, and knowledge of nuance and subtext encoded in language. **Davies (2025)** (arXiv:2504.12187) further argues that certain architectural features of LLMs satisfy the constraints of semantic description, syntactic structure, and causal systematicity required for tacit knowledge under Martin Davies's framework.

In the domain of process knowledge, **Gassen et al. (2025)** propose LLM-based chatbots that guide users through adaptive, interactive interviews to collect explicit knowledge and uncover tacit knowledge, validating the approach empirically in business process contexts.

**Anwar et al. (2022)** specifically address tacit-knowledge-based requirements elicitation in the COVID-19 context, demonstrating that structured elicitation protocols can surface domain expertise that standard requirements engineering misses -- a direct precedent for using structured QA scaffolding to surface tacit knowledge from interviews.

---

## 2. LLM-Assisted Analysis of Interviews and Narratives

A parallel stream investigates LLMs as tools for qualitative analysis of interview and narrative data -- the same data type that G-Transcriber processes.

**De Paoli (2024)** experiments with GPT-3.5-Turbo to perform inductive Thematic Analysis (TA) on semi-structured interviews, applying Braun and Clarke's six-phase framework. Results show the model can partially reproduce main themes, establishing feasibility of LLM-assisted qualitative analysis. **Dai et al. (2023)** formalize this as the "LLM-in-the-loop" paradigm for thematic analysis, where human researchers iteratively refine LLM-generated themes rather than performing manual coding from scratch.

**Liu & Sun (2025)** explore GPT-4 integration with human expertise for text analysis of stakeholder interviews on education policy. Their mixed-methods approach achieves 77.89% alignment with human coding for specific themes and 96.02% for broader themes, surpassing traditional NLP methods by over 25%. **Kwak et al. (2025)** compare neuro-symbolic and LLM-based information extraction on agricultural interview transcripts, with the LLM system achieving F1 of 69.4 vs. 52.7 for the rule-based approach.

For oral history specifically -- the closest analogue to ethnographic interviews -- **Cherukuri et al. (2025)** develop a framework to automate semantic and sentiment annotation for oral history archives (Japanese American Incarceration Oral Histories, Densho Digital Repository). ChatGPT achieved 88.71% F1 for semantic classification, with the pipeline scaled to 92,191 sentences from 1,002 interviews. **Schroeder et al. (2025)** provide a broader critical perspective on LLMs in qualitative research, examining uses, tensions, and intentions that arise when computational tools meet interpretive traditions.

---

## 3. Bloom's Taxonomy and Cognitively-Scaffolded QA Generation

The CEP (Cognitive Elicitation Pipeline) grounds QA generation in Bloom's Revised Taxonomy (Anderson & Krathwohl, 2001). Several recent works explore this intersection.

**Scaria et al. (2024)** examine five LLMs generating educational questions at different Bloom levels, using zero-shot, few-shot, and chain-of-thought prompting. A key finding is that an eight-shot technique improves alignment with the taxonomy, though higher-order levels (Evaluate, Create) remain difficult for all models. **Weidlich et al. (2024)** introduce BloomLLM, a fine-tuned ChatGPT-3.5-turbo trained on 1,026 taxonomy-annotated questions spanning 29 topics, demonstrating that supervised fine-tuning significantly outperforms zero-shot prompting for cognitive-level alignment.

**Bahl et al. (2024)** compare prompting strategies for generating and classifying questions within a 2D matrix framework encompassing multiple question types and cognitive levels. They find that alignment improves with model size and reasoning prompts, especially at lower cognitive levels (Remember, Understand). **Yuan et al. (2024)** propose PFQS (Planning First, Question Second), a two-stage approach where the LLM first generates an answer plan (reasoning steps and knowledge needed) and then generates questions requiring those planned steps -- enabling fine-grained control over question complexity, directly relevant to scaffolded generation.

---

## 4. LLM-as-a-Judge Evaluation Frameworks

G-Transcriber employs LLM-as-a-Judge for both QA validation (Phase 2) and Knowledge Coverage scoring (Phase 4). This paradigm has become a mature evaluation methodology.

**Zheng et al. (2023)** establish the foundational LLM-as-a-Judge paradigm with MT-Bench and Chatbot Arena, demonstrating that GPT-4 matches human preferences with over 80% agreement. They identify and characterize key biases: position bias, verbosity bias, and self-enhancement bias. **RAGAS (Es et al., 2024)** introduces a reference-free evaluation framework for RAG pipelines assessing faithfulness, answer relevance, and context relevance, becoming a de facto standard in RAG evaluation.

**ARES (Saad-Falcon et al., 2024)** extends RAGAS by generating synthetic training data to fine-tune lightweight LM judges, using prediction-powered inference (PPI) with small human annotation sets to correct systematic biases. ARES judges achieve substantially higher precision across eight knowledge-intensive tasks (KILT, SuperGLUE). **RAGEval (Zhu et al., 2025)** presents a schema-based framework for generating evaluation datasets across diverse domains, proposing three metrics -- Completeness, Hallucination, and Irrelevance -- with high consistency between LLM and human scoring.

**GroUSE (Muller et al., 2025)** provides a critical meta-evaluation lens: a benchmark of 144 unit tests that identifies seven specific failure modes in LLM judges for grounded QA. Their key finding -- that correlation with GPT-4 is an insufficient proxy for practical judge quality -- is essential for understanding the validity boundaries of LLM-as-a-Judge approaches.

---

## 5. Self-Containedness and Decontextualization

The CEP requires QA pairs to be self-contained for downstream GraphRAG evaluation. This connects to a growing literature on decontextualization.

**Choi et al. (2021)** define the foundational decontextualization task: rewriting sentences to be interpretable in isolation while preserving meaning. They identify four editing operations (pronoun-to-NP swap, discourse marker removal, scope bridging, information addition) and confirm that decontextualized sentences are more valuable as QA answers. **Gunjal & Durrett (2024)** introduce "molecular facts" -- the right granularity between fully atomic and overly coarse claims -- with two desiderata: decontextuality and minimality. This framework directly informs the balance between self-containedness and information preservation in QA generation.

**Newman et al. (2023)** propose QaDecontext, a three-stage framework (question generation, question answering, rewriting) for decontextualizing snippets from scientific documents. Their key insight -- that identifying what context is missing is harder than rewriting once identified -- validates the prompt-first approach adopted by the CEP, where negative constraints prevent context-dependent generation at the source rather than requiring post-hoc correction.

---

## 6. Knowledge Graph Construction from Text with LLMs

Phase 3 employs AutoSchemaKG for schema-free knowledge graph construction. This falls within a rapidly expanding literature on LLM-empowered KGC.

**Bian (2025)** provides a comprehensive survey examining how LLMs transform knowledge graph construction, analyzing the three-stage pipeline (ontology engineering, knowledge extraction, knowledge fusion) through schema-based and schema-free paradigms. Methods reviewed include ChatIE (extraction as multi-turn dialogue), KGGEN (sequential LLM invocations), and Retrieval-Augmented prompting.

**Bai et al. (2025)** introduce AutoSchemaKG itself -- the framework adopted by G-Transcriber -- which performs autonomous schema induction from unstructured text through an Extract-Define-Canonicalize pipeline. The key innovation is dynamic schema induction under the Open World Assumption: rather than constraining extraction to a fixed ontology, the LLM extracts triples freely and induces the schema from the data, producing a type system that reflects the actual conceptual structure. Event-aware extraction retains over 90% of informational content versus approximately 70% for entity-only approaches.

**Ji et al. (2022)** survey knowledge graphs broadly -- representation, acquisition, and applications -- providing the foundational context for KG construction methods. **Meher et al. (2025)** introduce LINK-KG, an LLM-driven approach to coreference-resolved knowledge graphs, addressing the entity resolution challenge that produces "knowledge islands" in document-level KGs -- directly relevant to the horizontal fragmentation observed in G-Transcriber's preliminary results.

**Li et al. (2025)** investigate mitigating LLM hallucinations with knowledge graphs, demonstrating the bidirectional relationship between KGs and LLMs: KGs ground LLM generation, while LLMs construct KGs. This reciprocity underpins the evaluation strategy where LLM-generated QA pairs test LLM-constructed graphs.

---

## 7. GraphRAG and Graph-Based Retrieval

The Knowledge Coverage evaluation in Phase 4 tests whether the KG supports question answering -- a core GraphRAG capability.

**Edge et al. (2024)** introduce GraphRAG (Microsoft Research), proposing a two-stage approach: (1) build an entity knowledge graph from source documents using an LLM, then (2) pre-generate community summaries for groups of closely related entities. At query time, partial responses from each community summary are aggregated. GraphRAG demonstrates substantial improvements over conventional RAG for global sensemaking questions over datasets in the 1-million-token range, particularly in comprehensiveness and diversity.

**Peng et al. (2024)** survey GraphRAG comprehensively -- downstream tasks, application domains, evaluation methodologies, and industrial use cases. They identify critical evaluation gaps: lack of domain-specific corpora and oversimplified task granularity. Their finding that rich KGs provide gains when balanced to avoid excess noise is relevant to the tradeoff between extraction recall and graph quality.

The application of KGs to oral historical archives is explored by **ACM (2024)** (Proceedings of the 8th International Conference on Information System and Data Mining), which uses LLM-RAG to construct knowledge graphs from oral historical archive resources, extracting time, location, characters, and events to form narrative records -- the closest published methodological parallel to G-Transcriber's interview-to-KG pipeline.

---

## 8. QA-Based Knowledge Graph Evaluation

Phase 4's functional evaluation -- using QA pairs as ground truth for KG assessment -- draws on emerging work in KG evaluation through question answering.

**KG-QAGen (Zhang et al., 2025)** extracts QA pairs at multiple complexity levels from structured representations along three dimensions: multi-hop retrieval, set operations, and answer plurality. The resulting 20,139 QA pairs (the largest long-context benchmark) reveal that even top models struggle with set-based comparisons and multi-hop inference. The multi-dimensional complexity framework is directly applicable to Bloom-stratified evaluation.

**Dynamic-KGQA (Dammu et al., 2025)** addresses data contamination in static KGQA benchmarks by generating new dataset variants on every run while preserving the underlying statistical distribution. This dynamic generation approach is valuable for iterative KG evaluation where the knowledge graph evolves.

**Zhou et al. (2022)** re-think knowledge graph completion evaluation from an information retrieval perspective, proposing IR-based metrics (MRR, Hits@k) as more robust alternatives to entity-ranking metrics for assessing KG quality. This IR lens connects KG evaluation to the TREC-PAR protocol proposed in the Etno-KGC project.

---

## 9. ASR Quality and Whisper

Phase 1 relies on Whisper for speech recognition. The ASR quality directly impacts all downstream processing.

**Radford et al. (2023)** introduce Whisper, trained on 680,000 hours of multilingual audio via large-scale weak supervision. The model demonstrates robust performance across languages and acoustic conditions, though quality varies significantly by language and recording environment. For Portuguese-Brazilian ethnographic recordings -- often in noisy field conditions with regional dialectal variation -- the quality validation pipeline implemented in G-Transcriber (script/charset match, repetition detection, segment patterns, content density) addresses known Whisper failure modes including hallucinated loops and language confusion.

---

## 10. Ethnographic Knowledge, AI, and Climate Resilience

The broader research context situates G-Transcriber within the intersection of AI, traditional ecological knowledge, and climate resilience -- a space characterized by a profound evidence gap.

### 10.1 Traditional Ecological Knowledge and Decolonial Frameworks

**Martin et al. (2010)** define Traditional Ecological Knowledge (TEK) as a cumulative body of knowledge, practice, and belief about the relationships of living beings with one another and their environment, evolving by adaptive processes and handed down through generations. This definition grounds what G-Transcriber aims to elicit and structure.

**Carroll et al. (2023)** formalize the **CARE Principles for Indigenous Data Governance** -- Collective Benefit, Authority to Control, Responsibility, and Ethics -- establishing the ethical framework for any AI system processing traditional community knowledge. The Etno-KGC project's commitment to participatory validation through TREC-PAR directly operationalizes these principles.

**BlackDeer (2023)** examines digital colonialism in the context of data and education, arguing that Indigenous data sovereignty requires decolonizing the instruments of data collection themselves. **Tapu & Fa'agau (2022)** explore AI as a potential instrument for decolonized data practices, identifying both the promise and the risks when computational tools engage with traditional knowledge systems.

### 10.2 The Evidence Gap

**Reckziegel & Costa (2025b)** -- the SLR companion to this project -- conduct a systematic literature review across eight academic databases at the intersection of AI, ancestral technologies, and climate resilience. Their central finding is the identification of a **profound evidence gap**: of 243 initial documents, the literature is characterized by "near-misses" -- articles that touch upon two of the three core pillars (AI + climate, community + climate, AI + community) but fail to integrate all three. This fragmentation across disciplinary silos validates the originality of the Etno-KGC approach, which bridges these domains through a unified technical pipeline grounded in decolonial principles.

### 10.3 Network Science Foundations

The topological analysis of the constructed knowledge graph draws on foundational network science concepts. **Barabasi & Posfai (2016)** provide the theoretical framework for scale-free networks and hub-and-spoke structures observed in the KG. **Fagiolo (2007)** formalizes clustering coefficients for directed networks, enabling the distinction between $C_{dir}$ and $C_{undir}$ that diagnoses hierarchical structure. **Humphries & Gurney (2008)** quantify small-world-ness, applicable to characterizing the navigability of sparse but locally clustered knowledge graphs. **Liben-Nowell & Kleinberg (2007)** address link prediction in social networks -- relevant to the future work of connecting "knowledge islands" through latent semantic triangles.

---

## 11. Positioning: What This Work Adds

The literature review reveals that while each component of G-Transcriber has precedents in isolation, their integration is novel:

1. **Tacit knowledge elicitation** has been explored in manufacturing (Rank et al., 2025) and organizational contexts (Zuin et al., 2025), but not from ethnographic interviews with climate-affected traditional communities.

2. **Bloom's Taxonomy QA generation** has been applied to educational content (Scaria et al., 2024; Weidlich et al., 2024), but not as a scaffolding mechanism for surfacing progressively deeper layers of tacit knowledge from oral narratives.

3. **Schema-free KG construction** via AutoSchemaKG (Bai et al., 2025) has been validated on web-scale corpora, but not on ethnographic interview transcriptions where the Open World Assumption is not merely a technical choice but an epistemological necessity -- the knowledge is inherently partial and emergent.

4. **KG evaluation via QA pairs** has been explored for structured domains (Zhang et al., 2025; Dammu et al., 2025), but the Bloom-stratified functional evaluation -- where the cognitive level of the question probes a specific layer of tacit knowledge representation -- is a novel contribution.

5. **The ethical-decolonial grounding** -- CARE Principles, participatory validation via TREC-PAR, data sovereignty -- is absent from the KGC and GraphRAG literature, which operates almost exclusively on anglophone, institutionally-produced corpora.

The SLR's evidence gap (Reckziegel & Costa, 2025b) confirms this positioning: the bridges between AI, ancestral knowledge, and climate resilience remain largely unbuilt. G-Transcriber does not merely apply existing techniques to a new domain; it constructs a pipeline where methodological choices at every stage are driven by the epistemological requirements of tacit knowledge in traditional communities.

---

## References

Anderson, L. W., & Krathwohl, D. R. (Eds.). (2001). *A Taxonomy for Learning, Teaching, and Assessing: A Revision of Bloom's Taxonomy of Educational Objectives*. Longman.

Anwar, H., Khan, S. U. R., Iqbal, J., & Akhunzada, A. (2022). A tacit-knowledge-based requirements elicitation model supporting COVID-19 context. *IEEE Access*, 10, 24481--24508.

Bahl, V., et al. (2024). Analysis of LLMs for educational question classification and generation. *Computers and Education: Artificial Intelligence*.

Bai, J., Fan, W., Hu, Q., Li, C., Tsang, H. T., Luo, H., Yim, Y., Huang, H., Zhou, X., et al. (2025). AutoSchemaKG: Autonomous knowledge graph construction through dynamic schema induction from web-scale corpora. *arXiv preprint arXiv:2505.23628*.

Barabasi, A., & Posfai, M. (2016). *Network Science*. Cambridge University Press.

Bian, H. (2025). LLM-empowered knowledge graph construction: A survey. *arXiv preprint arXiv:2510.20345*.

BlackDeer, A. A. (2023). Decolonizing Data; Indigenizing Education for Climate Action. *pressbooks.pub*.

Carroll, S. R., Garba, I., Figueroa-Rodriguez, O. L., Holbrook, J., Lovett, R., Materechera, S., Parsons, M., Raseroka, K., Rodriguez-Lonebear, D., Rowe, R., et al. (2023). *The CARE Principles for Indigenous Data Governance*. Open Scholarship Press.

Carroll, S. R., Herczog, E., Hudson, M., Russell, K., & Stall, S. (2021). Operationalizing the CARE and FAIR principles for indigenous data futures. *Scientific Data*, 8(1), 108.

Cherukuri, K. S., Moses, P. A., Sakata, A., Chen, J., & Chen, H. (2025). Large language models for oral history understanding with text classification and sentiment analysis. *arXiv preprint arXiv:2508.06729*.

Choi, E., Palomaki, J., Lamm, M., Kwiatkowski, T., Das, D., & Collins, M. (2021). Decontextualization: Making sentences stand-alone. *Transactions of the Association for Computational Linguistics*, 9, 447--461.

Dai, S.-C., Xiong, A., & Ku, L.-W. (2023). LLM-in-the-loop: Leveraging large language model for thematic analysis. *Findings of the Association for Computational Linguistics: EMNLP 2023*, 9993--10001.

Dammu, P. P. S., et al. (2025). Dynamic-KGQA: A scalable framework for generating adaptive question answering datasets. *Proceedings of SIGIR 2025*.

Davies, M. (2025). What do large language models know? Tacit knowledge as a potential causal-explanatory structure. *Philosophy of Science*. arXiv:2504.12187.

De Paoli, S. (2024). Performing an inductive thematic analysis of semi-structured interviews with a large language model. *Social Science Computer Review*, 42(4), 997--1019.

Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., Metropolitansky, D., Ness, R. O., & Larson, J. (2024). From local to global: A graph RAG approach to query-focused summarization. *arXiv preprint arXiv:2404.16130*.

Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024). RAGAS: Automated evaluation of retrieval augmented generation. *Proceedings of EACL 2024 (System Demonstrations)*.

Fagiolo, G. (2007). Clustering in complex directed networks. *Physical Review E*, 76, 026107.

Gassen, J. B., et al. (2025). Large language models for process knowledge acquisition. *Business & Information Systems Engineering*.

Gunjal, A., & Durrett, G. (2024). Molecular facts: Desiderata for decontextualization in LLM fact verification. *Findings of EMNLP 2024*.

Humphries, M. D., & Gurney, K. (2008). Network 'small-world-ness': A quantitative method for determining canonical network equivalence. *PLOS ONE*, 3(4), 1--10.

Ji, S., Pan, S., Cambria, E., Marttinen, P., & Yu, P. S. (2022). A survey on knowledge graphs: Representation, acquisition, and applications. *IEEE Transactions on Neural Networks and Learning Systems*, 33(2), 494--514.

Kwak, A. S., Alexeeva, M., Hahn-Powell, G., Alcock, K., McLaughlin, K., McCorkle, D., McNunn, G., & Surdeanu, M. (2025). Information extraction from conversation transcripts: Neuro-symbolic vs. LLM. *arXiv preprint arXiv:2510.12023*.

Li, H., Appleby, G., Alperin, K., Gomez, S. R., & Suh, A. (2025). Mitigating LLM hallucinations with knowledge graphs: A case study.

Liben-Nowell, D., & Kleinberg, J. (2007). The link-prediction problem for social networks. *Journal of the American Society for Information Science and Technology*, 58(7), 1019--1031.

Liu, A., & Sun, M. (2025). From voices to validity: Leveraging large language models (LLMs) for textual analysis of policy stakeholder interviews. *AERA Open*. arXiv:2312.01202.

Lu, J. (2025). Tacit knowledge in large language models. *The Review of Austrian Economics*.

Martin, J. F., Roy, E. D., Diemont, S. A., & Ferguson, B. G. (2010). Traditional ecological knowledge (TEK): Ideas, inspiration, and designs for ecological engineering. *Ecological Engineering*, 36(7), 839--849.

Meher, D., Domeniconi, C., & Correa-Cabrera, G. (2025). LINK-KG: LLM-driven coreference-resolved knowledge graphs for human smuggling networks.

Muller, S., Loison, A., Omrani, B., & Viaud, G. (2025). GroUSE: A benchmark to evaluate evaluators in grounded question answering. *Proceedings of COLING 2025*.

Newman, B., Soldaini, L., Fok, R., Cohan, A., & Lo, K. (2023). A question answering framework for decontextualizing user-facing snippets from scientific documents. *Proceedings of EMNLP 2023*.

Peng, B., Zhu, Y., Liu, Y., et al. (2024). Graph retrieval-augmented generation: A survey. *arXiv preprint arXiv:2408.08921*; *ACM Transactions on Information Systems* (2025).

Polanyi, M., & Sen, A. (2009). *The Tacit Dimension*. University of Chicago Press.

Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust speech recognition via large-scale weak supervision. *Proceedings of ICML*.

Rank, Y., Streloke, L., Brundl, P., Bodendorf, F., & Franke, J. (2025). Large language models for tacit knowledge elicitation in Industry 5.0: A literature review. *The Human Side of Service Engineering, AHFE Open Access Journal*, 182.

Reckziegel, F., & Costa, R. (2025a). Etno-KGC: Construcao de grafos de conhecimento em contextos etnograficos. *Projeto de Pesquisa, UFRGS/UFPel*.

Reckziegel, F., & Costa, R. (2025b). AI, ancestral technologies, and climate resilience in traditional communities: A systematic literature review. *Working paper, UFRGS/UFPel*.

Saad-Falcon, J., Khattab, O., Potts, C., & Zaharia, M. (2024). ARES: An automated evaluation framework for retrieval-augmented generation systems. *Proceedings of NAACL 2024*.

Scaria, N., Chenna, S. D., & Mishra, D. (2024). Automated educational question generation at different Bloom's skill levels using large language models: Strategies and evaluation. *ECTEL 2024*. arXiv:2408.04394.

Schroeder, H., Quere, M. A. L., Randazzo, C., Mimno, D., & Schoenebeck, S. (2025). Large language models in qualitative research: Uses, tensions, and intentions.

Tapu, I. F., & Fa'agau, T. K. (2022). A new age indigenous instrument: Artificial intelligence and its potential for (de)colonialized data. *Harvard Civil Rights-Civil Liberties Law Review*, 57(2), 715--753.

Weidlich, D., et al. (2024). BloomLLM: Large language models based question generation combining supervised fine-tuning and Bloom's taxonomy. *Proceedings of ECTEL 2024*, Springer.

Yuan, X., et al. (2024). Planning first, question second: An LLM-guided method for controllable question generation. *Findings of ACL 2024*.

Zhang, Y., et al. (2025). KG-QAGen: A knowledge-graph-based framework for systematic question generation and long-context LLM evaluation. *arXiv preprint arXiv:2505.12495*.

Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems*.

Zhou, Y., Chen, X., He, B., Ye, Z., & Sun, L. (2022). Re-thinking knowledge graph completion evaluation from an information retrieval perspective. *Proceedings of the 45th International ACM SIGIR Conference*, 916--926.

Zhu, K., Luo, Y., Xu, D., Wang, R., et al. (2025). RAGEval: Scenario specific RAG evaluation dataset generation framework. *Proceedings of ACL 2025*.

Zuin, G., Mastelini, S., Loures, T., & Veloso, A. (2025). Leveraging large language models for tacit knowledge discovery in organizational contexts. *IJCNN 2025*. arXiv:2507.03811.
