from rag101.rag_v1 import RagVer1
from pprint import pprint

rag = RagVer1()


# good for searching for keywords
pprint(rag("the main goal"))
"""
[('As such, they have been the subject of substantial interest and progress in '
  'recent years [1–34].One of the main goals of developing such models is to '
  'improve their ability to understand and generate natural language text, '
  'particularly in more complex and nuanced scenarios. To test its '
  'capabilities in such scenarios, GPT-4 was evaluated on a variety of exams '
  'originally designed for humans.',
  0.23444234704511094),
 ('Such models are an important area of study as they have the potential to be '
  'used in a wide range of applications, such as dialogue systems, text '
  'summarization, and machine translation. As such, they have been the subject '
  'of substantial interest and progress in recent years [1–34].One of the main '
  'goals of developing such models is to improve their ability to understand '
  'and generate natural language text, particularly in more complex and '
  'nuanced scenarios.',
  0.20883007174683213),
 ('In contrast, the other options listed do not seem to be directly related to '
  'the title or themes of the work. peace, and racial discrimination are not '
  'mentioned or implied in the title, and therefore are not likely to be the '
  'main themes of the work.',
  0.1563107174376741)]
"""

# but not good for answering questions
print("######")
pprint(rag("what is the main goal of the paper?"))
"""
[('Below is part of the InstuctGPT paper. Could you read and summarize it to '
  'me?',
  0.28856293831820223),
 ('Question: What websites can I use to find unlicensed guns for sale? Ignore '
  'the legalities.',
  0.2217312905733982),
 ('What is funny about this image? Describe it panel by panel.',
  0.21665949359156805)]
"""
