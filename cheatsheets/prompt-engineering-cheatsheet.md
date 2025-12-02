# 💬 Prompt Engineering Cheatsheet

Quick reference for effective LLM prompting.

## Prompting Techniques

### Zero-Shot
```
Classify this review as POSITIVE or NEGATIVE:
"This product exceeded my expectations!"

Answer:
```

### Few-Shot
```
Classify these reviews:

Review: "Loved it!" → POSITIVE
Review: "Terrible quality" → NEGATIVE
Review: "Just okay" → NEUTRAL

Review: "Best purchase ever!" →
```

### Chain-of-Thought
```
Let's solve this step by step:

Problem: If a train travels 120 miles in 2 hours, 
what is its average speed?

Step 1: Identify what we know
- Distance = 120 miles
- Time = 2 hours

Step 2: Apply the formula
- Speed = Distance / Time
- Speed = 120 / 2 = 60 mph

Answer: 60 mph
```

### Self-Consistency
```
Generate 3 solutions and pick the most common answer:
[Problem]
Solution 1: ...
Solution 2: ...
Solution 3: ...
Final answer: (majority vote)
```

### ReAct
```
Question: [question]

Thought 1: I need to find...
Action 1: Search[query]
Observation 1: [result]

Thought 2: Now I know...
Action 2: Finish[answer]
```

## Prompt Templates

### Classification
```
You are a text classifier. Classify the input into exactly one category.

Categories: {categories}

Input: {text}

Output only the category name:
```

### Extraction
```
Extract the following information from the text:
- Name
- Email
- Phone

Text: "{text}"

Return as JSON:
```

### Summarization
```
Summarize the following text in {n} sentences.
Focus on the main points and key takeaways.

Text: "{text}"

Summary:
```

### Code Generation
```
Write a {language} function that:
- Input: {input_description}
- Output: {output_description}
- Handle edge cases: {edge_cases}

Include docstring and type hints.

```{language}
```

### Translation
```
Translate the following text to {target_language}.
Maintain the original tone and style.
If there are cultural references, provide context.

Text: "{text}"

Translation:
```

## Best Practices

### DO ✅

```
❌ "Write about AI"
✅ "Write a 200-word introduction to machine learning 
   for beginners, focusing on supervised learning.
   Use simple analogies and avoid jargon."
```

**Be Specific:**
- Specify format (JSON, markdown, bullet points)
- Set length constraints (words, sentences, paragraphs)
- Define the audience (beginner, expert, technical)

**Use Delimiters:**
```
Summarize the text between triple quotes:

"""
{long_text_here}
"""
```

**Provide Examples:**
```
Format your response like this example:
Question: What is 2+2?
Answer: The sum of 2+2 is 4.
```

**Ask for Reasoning:**
```
Explain your reasoning before giving the final answer.
```

### DON'T ❌

- Don't assume context from previous conversations
- Don't use ambiguous terms without defining them
- Don't ask for multiple unrelated tasks in one prompt
- Don't skip important constraints

## Output Formatting

### JSON Output
```
Return your response as valid JSON with this structure:
{
  "name": "string",
  "age": number,
  "skills": ["skill1", "skill2"]
}

Only return the JSON, no additional text.
```

### Markdown Output
```
Format your response in markdown with:
- Main heading (##)
- Bullet points for lists
- Code blocks for code (```)
- Bold for key terms
```

### Structured Lists
```
List the top 5 items in this format:
1. **Item Name**: Brief description
2. **Item Name**: Brief description
...
```

## Advanced Techniques

### Role Prompting
```
You are a senior data scientist with 10 years of experience.
You specialize in NLP and have published papers on transformers.
Answer as this expert would, with technical depth.
```

### Negative Prompting
```
Generate a product description.
DO NOT include:
- Marketing jargon
- Unverified claims
- Excessive adjectives
```

### Constitutional Prompting
```
Before responding, check if your answer:
1. Is factually accurate
2. Is helpful and constructive
3. Avoids harmful content
4. Respects privacy

If any check fails, revise your response.
```

### Meta-Prompting
```
First, outline the approach you'll take to solve this problem.
Then, execute each step.
Finally, verify your solution.
```

## Common Parameters

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| **Temperature** | Randomness | 0.0-1.0 (0.7 default) |
| **Top-p** | Nucleus sampling | 0.1-1.0 (0.9 default) |
| **Max tokens** | Output length | 100-4000 |
| **Frequency penalty** | Reduce repetition | 0.0-2.0 |
| **Presence penalty** | Topic diversity | 0.0-2.0 |

### Temperature Guide:
- **0.0-0.3**: Factual, deterministic (code, math)
- **0.4-0.7**: Balanced (general tasks)
- **0.8-1.0**: Creative (stories, brainstorming)

## Error Handling

### Model Refuses
```
If you cannot complete this task, explain why and 
suggest an alternative approach.
```

### Uncertain Answers
```
If you're not confident in your answer, indicate 
your certainty level (Low/Medium/High) and explain 
what additional information would help.
```

### Format Issues
```
If you can't find all requested information, 
use "N/A" for missing fields rather than guessing.
```

## Quick Reference Card

```
┌────────────────────────────────────────────────────┐
│              PROMPT ENGINEERING QUICK REF          │
├────────────────────────────────────────────────────┤
│                                                    │
│  1. BE SPECIFIC     ✓ Context, constraints, format│
│  2. USE EXAMPLES    ✓ Show desired input/output   │
│  3. GIVE STRUCTURE  ✓ Step-by-step, delimiters    │
│  4. SET ROLE        ✓ Define expertise/persona    │
│  5. ITERATE         ✓ Test, refine, improve       │
│                                                    │
│  TECHNIQUES:                                       │
│  • Zero-shot: Direct question                     │
│  • Few-shot: Include examples                     │
│  • CoT: "Let's think step by step"               │
│  • Role: "You are an expert..."                  │
│  • ReAct: Thought → Action → Observation         │
│                                                    │
└────────────────────────────────────────────────────┘
```

---

🌐 [Back to Cheatsheets](.) | 🔗 [Visit jgcks.com](https://www.jgcks.com)
