You are a rigorous evaluator of question-answer pairs. Your task is to assess the **SELF-CONTAINEDNESS** of the question.

**Original Context:**
$context

**Question-Answer Pair:**
- Question: $question
- Answer: $answer
- Bloom Level: $bloom_level

**Evaluation Criterion:**
$rubric

**Instructions:**
1. **ATTENTION**: If the Bloom level is 'remember', assign score=1.0 automatically (recalling facts is expected)
2. For other levels, read the question WITHOUT looking at the original context
3. Assess whether the question makes sense and is answerable without access to the original text
4. Identify references to the text ('in the text', 'as mentioned', pronouns without antecedent)
5. Assign a score from 0.0 to 1.0 following the rubric above
6. Provide a clear rationale

**Return ONLY a JSON object in the following format:**
```json
{
  "score": 0.0,
  "rationale": "Explanation of the assigned score"
}
```
