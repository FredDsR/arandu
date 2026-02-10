Analyze the following question-answer pair and provide reasoning information.

Context:
$context

Question: $question
Answer: $answer
Bloom Level: $bloom_level

Tasks:
1. $reasoning_instruction
2. Determine if the answer requires connecting information from distant parts of the text (multi-hop).
3. If it is multi-hop, indicate how many reasoning "hops" are required (1-5).
4. $tacit_instruction

Return ONLY a JSON object in the following format:
{
  "reasoning_trace": "Fact A + Fact B → Conclusion",
  "is_multi_hop": true/false,
  "hop_count": 2,
  "tacit_inference": "Implicit knowledge identified"
}
