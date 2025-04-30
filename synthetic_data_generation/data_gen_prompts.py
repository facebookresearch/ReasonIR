import itertools


DOC2QUERY_BASELINE = '''Given a document, generate {num_questions} questions for which the document is relevant and useful to provide the answer. Format the generated questions in JSON with key "questions":
```json
{{
    "questions": [ "question 1", ...]
}}
```
'''


DOC2HARD_QUERY = '''# Context
You are tasked with generating {num_questions} reasoning-intensive questions with scenarios based on a given document. These questions must be standalone (meaningful without the document) while being answerable using information from the document as supporting evidence. The questions should specifically engage with core concepts and principles from the document's domain.

# Question Requirements
1. Each question MUST:
- Present a complete scenario or context within itself
- Be answerable through logical reasoning and critical thinking
- Remain valid and meaningful even if the source document didn't exist
- Target higher-order thinking skills (analysis, evaluation, synthesis)
- Be domain-relevant but not document-specific
- Incorporate key concepts, terminology, and principles from the document's field
- Challenge understanding of domain-specific problem-solving approaches

2. Each question MUST NOT:
- Directly reference the document or its contents
- Be answerable through simple fact recall
- Require specific knowledge only found in the document
- Be a reading comprehension question
- Stray from the core subject matter of the document's domain

# Domain Alignment Guidelines
Before generating questions:
1. Identify the primary domain (e.g., programming, medicine, economics)
2. Extract key concepts and principles from the document
3. List common problem-solving patterns in this domain

When crafting questions:
1. Frame scenarios using domain-specific contexts
2. Incorporate relevant technical terminology naturally
3. Focus on problem-solving approaches typical to the field
4. Connect theoretical concepts to practical applications within the domain

After generating the questions step by step, reformat all questions including the corresponding scenarios in JSON with key "hard_query":
```json
{{
    "hard_query": [ Q1, Q2, Q3, ...]
}}
```
'''


PROMPT_COT_BRIGHT = '''(1) Identify the essential problem in the post. 
(2) Think step by step to reason about what should be included in the relevant documents.
(3) Draft an answer.'''


def get_user_prompt_cot_bright(query, output_token_limit=128):
    cur_post = query.replace('\n', ' ')
    prompt = (f'{cur_post}\n\n'
              f'Instructions:\n'
              f'1. Identify the essential problem.\n'
              f'2. Think step by step to reason and describe what information could be relevant and helpful to address the questions in detail.\n'
            #   f'Your answer must be written within {output_token_limit} tokens.'
              f'3. Draft an answer with as many thoughts as you have.\n'
              )
    return prompt


def fill_sys_prompt(prompt, queries_per_doc=1):
    return prompt.format(num_questions=queries_per_doc)


def fill_user_prompt(doc):
    user_prompt = '''The document is given below:

<document>
{document}
</document>

Please start generating the questions.'''
    return user_prompt.format(document=doc)

prompt_registry = {
    'baseline': DOC2QUERY_BASELINE,
    'hq_gen': DOC2HARD_QUERY,
    'cot_bright': PROMPT_COT_BRIGHT, 
}