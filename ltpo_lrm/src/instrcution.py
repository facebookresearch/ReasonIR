from example import Example

query_instruction_bright = {
    "inst-l": "Represent the Biology post for retrieving relevant paragraphs: ",
    "bge": "Represent this Biology post for searching relevant passages: ",
    "inst-xl": "Represent the Biology post for retrieving relevant paragraphs: ",
    "e5": "Instruct: Given a Biology post, retrieve relevant passages that help answer the post\nQuery: ",
}

doc_instruction_bright = {
    "inst-l": "Represent the Biology paragraph for retrieval: ",
    "inst-xl": "Represent the Biology paragraph for retrieval: "
}
inst_q = {
        "Stack-Biology": "Represent the Biology post for retrieving relevant paragraphs: ",
        "Stack-Medical": "Represent the Medical Science post for retrieving relevant paragraphs: ",
        "Stack-Bioinformatics": "Represent the Bioinformatics post for retrieving relevant paragraphs: ",
        "MedXpertQA-Exam": "Represent the Medical Exam for retrieving relevant paragraphs that help answer the exam: ",
        "MedQA-Diag": "Represent the Medical Exam for retrieving relevant paragraphs that help answer the exam: ",
        "PMC-Treat": "Represent the Clinical Case for retrieving relevant paragraphs that help answer the case: ",
        "PMCPatients": "Represent the Clinical Case for retrieving similar cases that help diagnose the case: ",
        "IIYiPatients-EN": "Represent the Clinical Case for retrieving similar cases that help diagnose the case: ",
}
inst_d = {
        "Stack-Biology": "Represent the Biology paragraph for retrieval: ",
        "Stack-Medical": "Represent the Medical Science paragraph for retrieval: ",
        "Stack-Bioinformatics": "Represent the Bioinformatics paragraph for retrieval: ",
        "MedXpertQA-Exam": "Represent the Medical paragraph for retrieval: ",
        "MedQA-Diag": "Represent the Medical paragraph for retrieval: ",
        "PMC-Treat": "Represent the Medical paragraph for retrieval: ",
        "PMCPatients": "Represent the Clinical Case for retrieval: ",
        "IIYiPatients-EN": "Represent the Clinical Case for retrieval: ",
}
e5_q = {
        "Stack-Biology": "Instruct: Given a Biology post, retrieve relevant passages that help answer the post\nQuery: ",
        "Stack-Medical": "Instruct: Given a Medical Science post, retrieve relevant passages that help answer the post\nQuery: ",
        "Stack-Bioinformatics": "Instruct: Given a Bioinformatics post, retrieve relevant passages that help answer the post\nQuery: ",
        "MedXpertQA-Exam": "Instruct: Given a Medical Exam, retrieve relevant passages that help answer the exam\nQuery: ",
        'MedQA-Diag': "Instruct: Given a Medical Exam, retrieve relevant passages that help answer the exam\nQuery: ",
        "PMC-Treat": "Instruct: Given a Clinical Case, retrieve relevant passages that help answer the case\nQuery: ",
        "PMCPatients": "Instruct: Given a Clinical Case, retrieve similar cases that help diagnose the case\nQuery: ",
        "IIYiPatients-EN": "Instruct: Given a Clinical Case, retrieve similar cases that help diagnose the case\nQuery: ",
}
bmr_q = {k:v.removeprefix("Instruct: ") for k, v in e5_q.items()}
bmr_q["BEIR_NFCorpus"] = "Given a question, retrieve relevant documents that best answer the question\nQuery: "
grit = {
    "Stack-Biology": "<|user|>\nGiven a Biology post, retrieve relevant passages that help answer the post\n<|embed|>\n",
    "Stack-Medical": "<|user|>\nGiven a Medical Science post, retrieve relevant passages that help answer the post\n<|embed|>\n",
    "Stack-Bioinformatics": "<|user|>\nGiven a Bioinformatics post, retrieve relevant passages that help answer the post\n<|embed|>\n",
    "MedXpertQA-Exam": "<|user|>\nGiven a Medical Exam, retrieve relevant passages that help answer the exam\n<|embed|>\n",
    'MedQA-Diag': "<|user|>\nGiven a Medical Exam, retrieve relevant passages that help answer the exam\n<|embed|>\n",
    "PMC-Treat": "<|user|>\nGiven a Clinical Case, retrieve relevant passages that help answer the case\n<|embed|>\n",
    "PMCPatients": "<|user|>\nGiven a Clinical Case, retrieve similar cases that help diagnose the case\n<|embed|>\n",
    "IIYiPatients-EN": "<|user|>\nGiven a Clinical Case, retrieve similar cases that help diagnose the case\n<|embed|>\n",
}
nv = e5_q
nv["BEIR_NFCorpus"] = "Instruct: Given a question, retrieve relevant documents that answer the question\nQuery: "
query_instruction = {
    "inst-l": inst_q,
    "inst-xl": inst_q,
    "bge": "Represent this sentence for searching relevant passages: ",
    "e5": e5_q,
    "bmr-410m": bmr_q,
    "bmr-2b": bmr_q,
    "bmr-7b": bmr_q,
    "grit": grit,
    "sfr": e5_q,
    "nv": nv,
}

doc_instruction = {
    "inst-l": inst_d,
    "inst-xl": inst_d,
    "bmr-410m": "Represent this passage\npassage: ",
    "bmr-2b": "Represent this passage\npassage: ",
    "bmr-7b": "Represent this passage\npassage: ",
    "grit": "<|embed|>\n",
    "nv": "",
}

hyde = {
    "Stack-Biology": "Please write a passage to help answer the Biology post.\nPost: {TEXT}\nPassage: ",
    "Stack-Medical": "Please write a passage to help answer the Medical Science post.\nPost: {TEXT}\nPassage: ",
    "Stack-Bioinformatics": "Please write a passage to help answer the Bioinformatics post.\nPost: {TEXT}\nPassage: ",
    "MedXpertQA-Exam": "Please write a passage to help answer the Medical Exam.\nExam: {TEXT}\nPassage: ",
    "MedQA-Diag": "Please write a passage to help answer the Medical Exam.\nExam: {TEXT}\nPassage: ",
    "PMC-Treat": "Please write a passage to help answer the Clinical Case.\nCase: {TEXT}\nPassage: ",
    "PMCPatients": "Please write a similar case to help diagnose the Clinical Case.\nCase: {TEXT}\nSimilar Case: ",
    "IIYiPatients-EN": "Please write a similar case to help diagnose the Clinical Case.\nCase: {TEXT}\nSimilar Case: ",
}
query2doc = {
    "Stack-Biology": "Write a passage that help answer the Biology post.\n\n###Examples:###\n{EXAMPLE}\n\n###Real Test:###\nPost: {TEXT}\nPassage: ".format(EXAMPLE=Example["Stack-Biology"], TEXT="{TEXT}"),
    "Stack-Medical": "Write a passage to help answer the Medical Science post.\n\n###Examples:###\n{EXAMPLE}\n\n###Real Test:###\nPost: {TEXT}\nPassage: ".format(EXAMPLE=Example["Stack-Medical"], TEXT="{TEXT}"),
    "Stack-Bioinformatics": "Write a passage to help answer the Bioinformatics post.\n\n###Examples:###\n{EXAMPLE}\n\n###Real Test:###\nPost: {TEXT}\nPassage: ".format(EXAMPLE=Example["Stack-Bioinformatics"], TEXT="{TEXT}"),
    "MedXpertQA-Exam": "Write a passage to help answer the Medical Exam.\n\n###Examples:###\n{EXAMPLE}\n\n###Real Test:###\nExam: {TEXT}\nPassage: ".format(EXAMPLE=Example["MedXpertQA-Exam"], TEXT="{TEXT}"),
    "MedQA-Diag": "Write a passage to help answer the Medical Exam.\n\n###Examples:###\n{EXAMPLE}\n\n###Real Test:###\nExam: {TEXT}\nPassage: ".format(EXAMPLE=Example["MedQA-Diag"], TEXT="{TEXT}"),
    "PMC-Treat": "Write a passage to help answer the Clinical Case.\n\n###Examples:###\n{EXAMPLE}\n\n###Real Test:###\nCase: {TEXT}\nPassage: ".format(EXAMPLE=Example["PMC-Treat"], TEXT="{TEXT}"),
    "PMCPatients": "Write a similar case to help diagnose the Clinical Case.\n\n###Examples:###\n{EXAMPLE}\n\n###Real Test:###\nCase: {TEXT}\nSimilar Case: ".format(EXAMPLE=Example["PMCPatients"], TEXT="{TEXT}"),
    "IIYiPatients-EN": "Write a similar case to help diagnose the Clinical Case.\n\n###Examples:###\n{EXAMPLE}\n\n###Real Test:###\nCase: {TEXT}\nSimilar Case: ".format(EXAMPLE=Example["IIYiPatients-EN"], TEXT="{TEXT}"),
}
lamer = {
"Stack-Biology": "Give a Biology post and its possible relevant passages (most of these passages are wrong). Please write a correct passage that help answer the post.\nJust output the correct passage, don't explanation.\n\nPost: {TEXT}\n\nPossible Relevant Passages: {PASSAGE}",
"Stack-Medical": "Give a Medical Science post and its possible relevant passages (most of these passages are wrong). Please write a correct passage that help answer the post.\nJust output the correct passage, don't explanation.\n\nPost: {TEXT}\n\nPossible Relevant Passages: {PASSAGE}",
"Stack-Bioinformatics": "Give a Bioinformatics post and its possible relevant passages (most of these passages are wrong). Please write a correct passage that help answer the post.\nJust output the correct passage, don't explanation.\n\nPost: {TEXT}\n\nPossible Relevant Passages: {PASSAGE}",
"MedXpertQA-Exam": "Give a Medical Exam and its possible relevant passages (most of these passages are wrong). Please write a correct passage that help answer the exam.\nJust output the correct passage, don't explanation.\n\nExam: {TEXT}\n\nPossible Relevant Passages: {PASSAGE}",
"MedQA-Diag": "Give a Medical Exam and its possible relevant passages (most of these passages are wrong). Please write a correct passage that help answer the exam.\nJust output the correct passage, don't explanation.\n\nExam: {TEXT}\n\nPossible Relevant Passages: {PASSAGE}",
"PMC-Treat": "Give a Clinical Case and its possible relevant passages (most of these passages are wrong). Please write a correct passage that help answer the case.\nJust output the correct passage, don't explanation.\n\nCase: {TEXT}\n\nPossible Relevant Passages: {PASSAGE}",
"PMCPatients": "Give a Clinical Case and its possible similar cases (most of these cases are non-similar). Please write a correct similar case that help diagnose the case.\nJust output the correct case, don't explanation.\n\nCase: {TEXT}\n\nPossible Similar Cases: {PASSAGE}",
"IIYiPatients-EN": "Give a Clinical Case and its possible similar cases (most of these cases are non-similar). Please write a correct similar case that help diagnose the case.\nJust output the correct case, don't explanation.\n\nCase: {TEXT}\n\nPossible Similar Cases: {PASSAGE}",
}
generate_hy_doc = {
    "hyde": hyde,
    "query2doc": query2doc,
    "lamer": lamer,
}

Check_Answer_accuracy_prompt = '''You are a medical QA evaluation expert. You will be given a medical question, its reference (gold) answer and a model-generated response (may include explanation).
Your task is to assess the correctness of the model's predicted answer based strictly on whether it matches the reference answer.

**Important Instructions:**
- 1.The reference answer is always 100% correct and authoritative.
- 2.You must not use any medical reasoning to judge whether the model's answer is reasonable.
- 3.Your job is only to determine whether the final answer extracted from the model response matches the reference answer.
- 4.Semantic matches (e.g., synonyms or medically equivalent terms) are acceptable, but different medical entities are not considered correct, even if they seem more appropriate.

**Evaluation Steps:**  
- 1.Extract the final answer (predicted entity or conclusion) from the model response.
- 2.Compare it to the reference answer and determine whether they match..
- 3.Provide an explanation for your judgment based only on answer alignment, not reasoning quality.


**Output Format:**  
Your output should follow the following format, do not output any additional content:
- Model answer: Extracted model answer
- Correctness: True or False
- Explanation: Detailed explanation

Here is the Q-A and model-generated response:
<QUESTION>
{QUESTION}
</QUESTION>
<Reference Answer>
{ANSWER}
</Reference Answer>

<Model Response>: 
{RESPONSE}
</Model Response>
'''