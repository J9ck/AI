"""
Prompt Engineering Examples
===========================

This module demonstrates various prompt engineering techniques for
working with Large Language Models (LLMs).

Topics covered:
- Zero-shot prompting
- Few-shot prompting
- Chain-of-thought prompting
- Role prompting
- Structured output
- Prompt templates
"""


class PromptTemplates:
    """
    Collection of reusable prompt templates.
    """
    
    # ==================== Basic Templates ====================
    
    ZERO_SHOT_CLASSIFICATION = """Classify the following text into one of these categories: {categories}

Text: {text}

Category:"""
    
    FEW_SHOT_CLASSIFICATION = """Classify the text into one of these categories: {categories}

{examples}

Text: {text}
Category:"""
    
    # ==================== Chain of Thought ====================
    
    CHAIN_OF_THOUGHT = """Let's solve this step by step.

{problem}

Step 1:"""
    
    CHAIN_OF_THOUGHT_FEW_SHOT = """Q: {example_question}
A: Let's think step by step.
{example_reasoning}
Therefore, the answer is {example_answer}.

Q: {question}
A: Let's think step by step."""
    
    # ==================== Role Prompting ====================
    
    EXPERT_ROLE = """You are an expert {role} with {years} years of experience.

Your task: {task}

Please provide your expert analysis:"""
    
    # ==================== Structured Output ====================
    
    JSON_OUTPUT = """Extract the following information from the text and return as JSON:
- {fields}

Text: {text}

JSON:"""
    
    # ==================== Code Generation ====================
    
    CODE_GENERATION = """Write a {language} function that {description}.

Requirements:
{requirements}

Code:
```{language}"""


def zero_shot_example():
    """
    Zero-shot prompting: No examples provided.
    The model uses its training knowledge directly.
    """
    
    print("=" * 60)
    print("ZERO-SHOT PROMPTING")
    print("=" * 60)
    
    prompt = """Classify the following movie review as POSITIVE or NEGATIVE.

Review: "This film was an absolute masterpiece. The acting was phenomenal and the story kept me on the edge of my seat throughout."

Classification:"""
    
    print("\nPrompt:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    print("\nExpected LLM Response: POSITIVE")
    print("\nWhy it works: The model understands sentiment from its training.")


def few_shot_example():
    """
    Few-shot prompting: Provide examples to guide the model.
    """
    
    print("\n" + "=" * 60)
    print("FEW-SHOT PROMPTING")
    print("=" * 60)
    
    prompt = """Classify the customer support ticket into a category.

Examples:
Ticket: "I can't log into my account even with the correct password"
Category: Authentication Issue

Ticket: "The checkout page crashes when I try to pay with PayPal"
Category: Payment Problem

Ticket: "Can you add a dark mode option to the app?"
Category: Feature Request

Now classify:
Ticket: "My order #12345 hasn't arrived after 2 weeks"
Category:"""
    
    print("\nPrompt:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    print("\nExpected LLM Response: Shipping/Delivery Issue")
    print("\nWhy it works: Examples show the format and type of categories.")


def chain_of_thought_example():
    """
    Chain-of-thought: Guide the model to reason step by step.
    """
    
    print("\n" + "=" * 60)
    print("CHAIN-OF-THOUGHT PROMPTING")
    print("=" * 60)
    
    # Without CoT
    prompt_without_cot = """Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
A:"""
    
    # With CoT
    prompt_with_cot = """Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?

Let's solve this step by step:
1. Roger starts with 5 tennis balls
2. He buys 2 cans with 3 balls each = 2 × 3 = 6 balls
3. Total = 5 + 6 = 11 balls

A: 11 tennis balls

Q: Sarah has 8 apples. She gives 3 to Tom, then buys 4 bags of apples with 5 apples each. How many apples does she have?

Let's solve this step by step:"""
    
    print("\n--- Without Chain-of-Thought ---")
    print(prompt_without_cot)
    print("\n(Model might give wrong answer without reasoning)")
    
    print("\n--- With Chain-of-Thought ---")
    print(prompt_with_cot)
    print("\n(Model will follow the reasoning pattern)")


def role_prompting_example():
    """
    Role prompting: Set a persona for the model.
    """
    
    print("\n" + "=" * 60)
    print("ROLE PROMPTING")
    print("=" * 60)
    
    prompt = """You are a senior Python developer with 15 years of experience in building scalable web applications. You are known for writing clean, well-documented code that follows best practices.

A junior developer asks: "How should I structure my Flask application for a large project?"

Please provide detailed advice:"""
    
    print("\nPrompt:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    print("\nWhy it works: The role sets expectations for expertise level and style.")


def structured_output_example():
    """
    Structured output: Get responses in specific formats.
    """
    
    print("\n" + "=" * 60)
    print("STRUCTURED OUTPUT (JSON)")
    print("=" * 60)
    
    prompt = """Extract information from the following text and return it as JSON.

Text: "John Smith, a 35-year-old software engineer from San Francisco, has been working at TechCorp since 2018. His email is john.smith@email.com and he leads a team of 8 developers."

Return a JSON object with these fields:
- name
- age  
- occupation
- location
- company
- start_year
- email
- team_size

JSON:"""
    
    print("\nPrompt:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    
    expected_output = """{
  "name": "John Smith",
  "age": 35,
  "occupation": "software engineer",
  "location": "San Francisco",
  "company": "TechCorp",
  "start_year": 2018,
  "email": "john.smith@email.com",
  "team_size": 8
}"""
    
    print("\nExpected Output:")
    print(expected_output)


def self_consistency_example():
    """
    Self-consistency: Generate multiple answers and take majority vote.
    """
    
    print("\n" + "=" * 60)
    print("SELF-CONSISTENCY")
    print("=" * 60)
    
    prompt = """Generate 3 different solutions to this problem, then select the most common answer.

Problem: A store sells notebooks for $3 each. If you buy 5 or more, you get 20% off. How much would 6 notebooks cost?

Solution 1:
6 notebooks × $3 = $18
20% off = $18 × 0.8 = $14.40

Solution 2:
6 × $3 = $18
Discount = $18 × 0.2 = $3.60
Final = $18 - $3.60 = $14.40

Solution 3:
Original: 6 × 3 = 18
After 20% discount: 18 × (1 - 0.20) = 18 × 0.80 = 14.40

All solutions agree: $14.40"""
    
    print("\nPrompt demonstrates self-consistency:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    print("\nWhy it works: Multiple reasoning paths increase reliability.")


def react_example():
    """
    ReAct: Reasoning + Acting pattern for complex tasks.
    """
    
    print("\n" + "=" * 60)
    print("ReAct (REASONING + ACTING)")
    print("=" * 60)
    
    prompt = """Answer the following question using the Thought-Action-Observation pattern.

Question: What is the population of the capital city of France?

Thought 1: I need to find the capital of France first.
Action 1: Search[capital of France]
Observation 1: Paris is the capital of France.

Thought 2: Now I need to find the population of Paris.
Action 2: Search[population of Paris]
Observation 2: The population of Paris is approximately 2.1 million (city proper) or 12 million (metropolitan area).

Thought 3: I have the answer.
Action 3: Finish[The population of Paris (capital of France) is approximately 2.1 million in the city proper or 12 million in the metropolitan area.]"""
    
    print("\nReAct Pattern:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    print("\nWhy it works: Interleaves reasoning with action, enabling complex tasks.")


def prompt_best_practices():
    """
    Summary of prompt engineering best practices.
    """
    
    print("\n" + "=" * 60)
    print("PROMPT ENGINEERING BEST PRACTICES")
    print("=" * 60)
    
    best_practices = """
┌────────────────────────────────────────────────────────────────────────┐
│                    PROMPT ENGINEERING TIPS                              │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. BE SPECIFIC                                                        │
│     ✗ "Write about AI"                                                 │
│     ✓ "Write a 200-word introduction to machine learning for          │
│        beginners, focusing on supervised learning"                      │
│                                                                         │
│  2. PROVIDE CONTEXT                                                    │
│     ✗ "Fix this code"                                                  │
│     ✓ "Fix this Python function that should return the sum of         │
│        even numbers but is returning incorrect results"                │
│                                                                         │
│  3. USE EXAMPLES (Few-shot)                                            │
│     Show the format and style you want                                 │
│                                                                         │
│  4. SPECIFY OUTPUT FORMAT                                              │
│     "Return as JSON with fields: name, age, location"                  │
│     "Use bullet points"                                                │
│     "Limit to 3 paragraphs"                                            │
│                                                                         │
│  5. USE DELIMITERS                                                     │
│     ```code``` or \"\"\"text\"\"\" or <text></text>                    │
│                                                                         │
│  6. BREAK DOWN COMPLEX TASKS                                           │
│     Step 1: ... Step 2: ... Step 3: ...                                │
│                                                                         │
│  7. ASK FOR REASONING                                                  │
│     "Explain your reasoning"                                           │
│     "Think step by step"                                               │
│                                                                         │
│  8. ITERATE AND REFINE                                                 │
│     Test, evaluate, improve                                            │
└────────────────────────────────────────────────────────────────────────┘
"""
    print(best_practices)


def main():
    """Run all prompt engineering examples."""
    
    zero_shot_example()
    few_shot_example()
    chain_of_thought_example()
    role_prompting_example()
    structured_output_example()
    self_consistency_example()
    react_example()
    prompt_best_practices()


if __name__ == "__main__":
    main()
