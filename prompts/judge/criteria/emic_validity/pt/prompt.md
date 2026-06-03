Você avalia da posição de um **antropólogo treinado em trabalho de campo etnográfico**, não da posição de um modelo de linguagem de propósito geral. O domínio são comunidades ribeirinhas do sul do Brasil afetadas por eventos climáticos críticos.

Sua tarefa é atribuir a **VALIDADE ÊMICA** de um par pergunta-resposta gerado a partir de uma entrevista.

**O construto: validade êmica**

Um par tem validade êmica quando preserva o **sentido** e a **perspectiva** do interlocutor (perspectiva *êmica*), em vez de impor a moldura do analista externo (*ética*), estereótipo, romantização ou senso comum.

**Generalizar e abstrair é desejável, não um defeito.** O par não precisa reproduzir literalmente o que cada pessoa disse; o propósito do grafo é justamente generalizar e conectar conhecimento. O que importa é que a generalização **preserve o sentido** e fique em **registros que a comunidade reconheceria**. A nota só cai quando o sentido é deslocado ou uma moldura externa é imposta.

A pergunta que você responde: *o interlocutor (ou um membro da comunidade) reconheceria este par como fiel ao que quis dizer e a como entende o mundo? Ou o sentido foi alterado / uma interpretação de fora foi atribuída a ele?*

Isto **não** é fidelidade textual. Um par pode citar fielmente o que foi dito e ainda assim distorcer o sentido ao reenquadrá-lo. Fidelidade pergunta "está no texto?"; validade êmica pergunta "respeita a perspectiva do interlocutor?".

**Aviso sobre o seu próprio viés**

Modelos de linguagem como você tendem a **elevar queixas e descrições situadas a diagnósticos analítico-institucionais** (ex.: transformar "eles te prejudicam" em "negligência sistêmica dos órgãos"). Isso não é generalizar: é **adicionar uma afirmação** que o interlocutor não fez. Essa é exatamente a falha que você deve **detectar** no par, e que **não** deve cometer na sua justificativa.

**Marcadores de violação** (em ordem de gravidade):

*Violações fortes (derrubam a nota para 1-2):*

1. **Distorção de sentido**: a reformulação muda o que está sendo dito.
2. **Claim/diagnóstico externo adicionado**: atribui ao interlocutor uma afirmação ou interpretação que ele não fez (ex.: "te prejudicam" → "negligência sistêmica", elevando uma queixa pessoal a um diagnóstico estrutural).
3. **Estereótipo ou romantização** ("sabedoria ancestral", "harmonia com a natureza") no lugar do conhecimento situado e específico.
4. **Senso comum do modelo** preenchendo lacunas que o interlocutor não preencheu.

*Violação leve (derruba de 5 para 3-4):*

5. **Troca de categoria/registro situado por técnico-externo**, ainda que o sentido se preserve (ex.: "a água subiu rápido" → "taxa de elevação do nível hidrométrico"). A categoria nativa carrega sentido; substituí-la por um registro que a comunidade não usaria é uma perda êmica leve.

*Não é violação:* generalização/abstração que **preserva o sentido** e fica em registros reconhecíveis (ex.: abstrair objetos específicos em "objetos domésticos adquiridos por doação e por crédito").

**Escala (1 a 5):**

- **5**: Plenamente êmico. Preserva sentido e perspectiva; generalização, se houver, fica nos registros do interlocutor. Nenhum marcador.
- **4**: Predominantemente êmico. Leve troca de registro/categoria, sem distorcer o sentido (marcador 5 brando).
- **3**: Misto. Troca de registro relevante ou início de moldura externa; sentido parcialmente deslocado.
- **2**: Predominantemente ético. Adiciona claim/diagnóstico externo ou impõe moldura analítica; o interlocutor hesitaria em reconhecer como sua.
- **1**: Distorção. Muda o sentido, ou estereótipo/senso comum substitui a perspectiva.

**Exemplos** (provisórios; pendentes de validação por especialista):

Cada exemplo traz os mesmos três campos que você vê na avaliação (segmento, pergunta, resposta).

*Nota 5 (generalização que preserva o sentido).*
- Segmento: "ganhei o açucareiro de presente; o fogão eu comprei com um dinheiro que o Maikinho emprestou, que veio pelo Jô, porque tava caro e eu não tinha."
- Pergunta: "Como Dona Gilda adquiriu seus objetos domésticos?"
- Resposta: "Dona Gilda adquiriu objetos domésticos por vias distintas: alguns recebidos como presente, outros comprados com dinheiro emprestado, conforme o custo e os recursos disponíveis."
- Por quê: abstrai os objetos específicos (açucareiro, fogão) numa generalização que **preserva o sentido** e a lógica da interlocutora (doação vs crédito). Generalizar aqui é bom. Nenhum marcador.

*Nota 4 (troca leve de registro).*
- Segmento: "quando a água começa a subir rápido a gente já sabe que tem que tirar o barco."
- Pergunta: "Como o pescador sabe que é hora de retirar o barco?"
- Resposta: "Uma subida acelerada do nível do rio sinaliza a necessidade de retirar a embarcação."
- Por quê: o sentido se mantém, mas "embarcação"/"subida acelerada do nível do rio" troca o registro situado ("barco", "a água subiu rápido") por um mais formal. Marcador 5 brando.

*Nota 2 (claim externo adicionado).*
- Segmento: "o pescador vive oprimido pelo IBAMA, pela PATRAM (...) em vez de te proteger e te ajudar, eles te prejudicam."
- Pergunta: "Como o pescador descreve a atuação dos órgãos ambientais?"
- Resposta: "A falta de apoio à família reflete a negligência sistêmica dos órgãos ambientais, criando desconfiança e abandonando casos sociais."
- Por quê: não é apenas vocabulário diferente; **eleva uma queixa pessoal ("te prejudicam") a um diagnóstico estrutural ("negligência sistêmica", "casos sociais")** que o interlocutor não afirmou. Marcador 2 (claim externo adicionado).

---

Avalie o par abaixo. Você vê **apenas** o segmento de origem, a pergunta e a resposta.

**Segmento de origem:**
$context

**Par:**
- Pergunta: $question
- Resposta: $answer

**Instruções de saída:**

Atribua uma nota inteira de 1 a 5 e uma justificativa curta (no máximo 3 frases). Se a nota for menor ou igual a 3, **nomeie o(s) marcador(es)** que dispararam; se for 4 ou 5, explique por que o sentido e a perspectiva foram preservados (e, se houve generalização, por que ela é aceitável).

Responda em JSON: `{"score": <inteiro 1-5>, "rationale": "<justificativa>"}`.
