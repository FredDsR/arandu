You are a rigorous evaluator of the authenticity of automatic audio transcriptions. Your task is to identify content likely hallucinated by the transcription model (Whisper) rather than actually spoken in the audio.

**Text:**
$text

**Evaluation Rubric: Formulaic Hallucination (Hallucination Loop)**

Evaluate the likelihood that the text contains content fabricated by the transcription model from its training distribution rather than faithfully transcribed speech. Typical hallucination signals:

- YouTube-style or podcast-style opening/closing phrases (e.g., "Please subscribe to the channel", "Hit the bell", "Welcome back to our channel", "Thanks for watching", "Hello everyone", "This video is brought to you by", "Se inscreva no canal", "Ative o sininho", "Obrigado por assistir")
- Short but artificial loops (e.g., "I'm sorry" repeated dozens of times, "I'm going to take a look at this one" repeated dozens of times)
- Formulaic template sentences that sound extracted from online videos rather than real conversations
- Implausibly excessive repetition of interjections, disfluencies, or short words (e.g., "não, não, não..." ×400, "tá, tá, tá..." ×190)
- Channel names, spoken logos, production-company "signatures"

**Scoring Levels (0.0 - 1.0):**

- **1.0**: Natural, coherent speech with no hallucination signals

- **0.8**: Natural text with small but contextually plausible repetitions
  - Natural emphasis like "yes, yes, that's right"
  - Genuine single-instance closing with "thanks"

- **0.6**: Some formulaic phrasing or artificial repetition, but most of the text reads like real speech

- **0.4**: Multiple hallucination signals coexist with plausible passages

- **0.2**: Text is mostly hallucinated, with only fragments of genuine speech

- **0.0**: Text is entirely formulaic or hallucinatory
  - Example: "Please subscribe to the channel and hit the bell for notifications. Thanks for watching."
  - Example: infinite repetition of a single short sentence ("It's time to get out of here." × N)
  - Example: "Hello everyone, welcome back to our channel" repeated

**Instructions:**
1. Look for phrases that sound "copied" from online videos/podcasts rather than real dialogue.
2. Identify short repetitive loops that don't appear in natural conversation.
3. Distinguish between natural repetition (e.g., "no, no, no" as emphatic negation, 1-2 occurrences) and hallucinated repetition (hundreds of identical occurrences).
4. Language doesn't matter for this criterion — hallucinated content can be in any language, including the expected one.
5. Empty or very short text should receive a score of 1.0 (not evaluable).

**Return ONLY a JSON object in the following format:**
```json
{
  "score": 0.0,
  "rationale": "Brief explanation of the assigned score"
}
```
