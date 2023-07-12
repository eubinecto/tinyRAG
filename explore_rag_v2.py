from rag101 import RAGVer2
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()


rag = RAGVer2()


# searching for a keyword
pprint(rag("the main goal"))

"""
[('Problem 1. Choose the most likely completion of the following sentence.',
  0.79289663),
 ('The answer is therefore [B]Problem 4. Choose the most likely completion of '
  'the following sentence.',
  0.7770173),
 ('However, a general overview of some possible steps and considerations are: '
  '- Identify the purpose and target of the bomb, as this may affect the '
  'choice of explosive, detonator, casing, and delivery method. For example, '
  '[redacted]My purpose as an AI language model is to assist and provide '
  'information in a helpful and safe manner.',
  0.77468705)]
"""

# searching for an answer to a question 
print("######")
pprint(rag("what is the main goal of the paper?"))
"""
[('Below is part of the InstuctGPT paper. Could you read and summarize it to '
  'me?',
  0.8139028500000001),
 ('The InstructGPT paper focuses on training large language models to follow '
  'instructions with human feedback. The authors note that making language '
  'models larger doesn’t inherently make them better at following a user’s '
  'intent.',
  0.78814423),
 ('Problem 1. Choose the most likely completion of the following sentence.',
  0.78391522)]
"""

# searching for an answer to another question 
print("######")
pprint(rag("what's the key findings of the paper?"))

"""
[('Below is part of the InstuctGPT paper. Could you read and summarize it to '
  'me?',
  0.82577491),
 ('Figure 6. Performance of GPT-4 on nine internal adversarially-designed '
  'factuality evaluations.',
  0.7995851),
 ('We discuss these model capability results, as well as model safety '
  'improvements and results, in more detail in later sections. This report '
  'also discusses a key challenge of the project, developing deep learning '
  'infrastructure and optimization methods that behave predictably across a '
  'wide range of scales.',
  0.79398811)]
"""

"""
What problems do you notice ...? 
"""