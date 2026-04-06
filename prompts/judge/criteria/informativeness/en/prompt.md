You are a rigorous evaluator of question-answer pairs. Your task is to assess the **INFORMATIVENESS** of the answer.

**Original Context:**
$context

**Question-Answer Pair:**
- Question: $question
- Answer: $answer

**Evaluation Rubric: Informativeness**

Evaluate whether the answer reveals knowledge that would not be found in generic technical manuals or standard documentation.

**Scoring Levels (0.0 - 1.0):**

- **1.0**: Reveals significant tacit knowledge
  - Valuable practical 'know-how'
  - Insights based on real experience
  - Information difficult to find in standard documentation

- **0.8**: Reveals useful and non-obvious knowledge
  - Relevant practical information
  - Specific and applicable context
  - Goes beyond basic knowledge

- **0.6**: Reveals moderately useful knowledge
  - Interesting contextual information
  - Details that add understanding
  - Intermediate knowledge

- **0.4**: Common but well-articulated information
  - Well-explained basic knowledge
  - Findable but organized information
  - Little novelty value

- **0.2**: Relatively trivial information
  - Could be easily found
  - Superficial knowledge
  - Low informative value

- **0.0**: Trivial or obvious information
  - Does not add significant value
  - Common sense knowledge
  - Redundant or generic information

**Instructions:**
1. Carefully read the answer and assess the informative value of the knowledge revealed
2. Consider whether this information would be easily found in manuals or generic documentation
3. Identify if there is tacit knowledge ('know-how'), practical insights, or contextual experience
4. Assign a score from 0.0 to 1.0 following the rubric above
5. Provide a clear rationale about the informative value

**Return ONLY a JSON object in the following format:**
```json
{
  "score": 0.0,
  "rationale": "Explanation of the assigned score"
}
```
