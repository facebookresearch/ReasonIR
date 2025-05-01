bright_aops = """We want to find different but similar math problems to the following problem:
{}
A document is relevant if it uses the same class of functions and shares **any** overlapping techniques.
Document: {}
Score the document above. The answer should be 'Relevance score: X.' where X is a number from 0-5.
0 means completely irrelevant, 5 means highly relevant and completely addresses the query. Don't output anything else.
"""

bright_theoremqa_questions = """We want to find a document which uses the same mathematical process as this one: 
{}
A document is relevant if it uses the same mathematical process as the query.
Document: {}
Score the document above. The answer should be 'Relevance score: X.' where X is a number from 0-5.
0 means completely irrelevant, 5 means highly relevant and completely addresses the query. Don't output anything else.
"""

bright_leetcode = """I am looking to find different problems that share similar data structures
(of any kind) or algorithms (e.g. DFS, DP, sorting, traversals, etc.). I am looking for problems that share one or both of these similarities to
this:
{}
Does the passage below share any similarities? e.g. if there was a textbook on leetcode problems, this would be in the same book even though it could be in a different chapter.
Passage: {}
Please rate the passage above. The answer should be 'Relevance score: X.' where X is a number from 0-5.
0 means completely irrelevant, 5 means highly relevant and completely addresses the query. Don't output anything else.
"""

bright_pony = """I will use the programming language pony. 
Problem: {}
But to solve the problem above, I need to know things about pony. A passage is relevant if it contains docs that match any part (even basic parts) of the code I will have to write for the above program.
Passage: {}
Please rate the passage above. The answer should be 'Relevance score: X.' where X is a number from 0-5.
0 means completely irrelevant, 5 means highly relevant and completely addresses the query. Don't output anything else.
"""

bright_theoremqa_theorems = """"We want to find a document which uses the same mathematical process as this one: 
{}
A document is relevant if it uses the same mathematical process as the query.
Document: {}
Score the document above. The answer should be 'Relevance score: X.' where X is a number from 0-5.
0 means completely irrelevant, 5 means highly relevant and completely addresses the query. Don't output anything else.
"""

bright_general = """A document is relevant if it contains information that helps answer or address the query.
A document is not relevant if it doesn't contain information that helps answer the query, even if it mentions similar topics.
Is the document below relevant to answering the query below?
The answer should be 'Relevance score: X.' where X is a number from 0-5.
0 means completely irrelevant, 5 means highly relevant and completely addresses the query. Don't output anything else.
Here is the query:
<start_query>
{}
<end_query>
Here is the document:
<start_document>
{}
<end_document>
"""
