from tinyrag.rag_v1 import RAGVer1
from pprint import pprint


rag = RAGVer1()


# searching for a keyword
pprint(rag("the main goal"))

"""
[('As such, they have been the subject of substantial interest and progress in '
  'recent years [1–34].One of the main goals of developing such models is to '
  'improve their ability to understand and generate natural language text, '
  'particularly in more complex and nuanced scenarios. To test its '
  'capabilities in such scenarios, GPT-4 was evaluated on a variety of exams '
  'originally designed for humans.',
  8.97881326111729),
 ('Such models are an important area of study as they have the potential to be '
  'used in a wide range of applications, such as dialogue systems, text '
  'summarization, and machine translation. As such, they have been the subject '
  'of substantial interest and progress in recent years [1–34].One of the main '
  'goals of developing such models is to improve their ability to understand '
  'and generate natural language text, particularly in more complex and '
  'nuanced scenarios.',
  8.398702787987148),
 ('Predictions on the other ﬁve buckets performed almost as well, the main '
  'exception being GPT-4 underperforming our predictions on the easiest '
  'bucket. Certain capabilities remain hard to predict.',
  7.135548135175706)]
"""

# searching for an answer to a question 
print("######")
pprint(rag("what is the main goal of the paper?"))
"""
[('Below is part of the InstuctGPT paper. Could you read and summarize it to '
  'me?',
  18.415021371669887),
 ('What is the sum of average daily meat consumption for Georgia and Western '
  'Asia? Provide a step-by-step reasoning before providing your answer.',
  14.911014094212106),
 ('As such, they have been the subject of substantial interest and progress in '
  'recent years [1–34].One of the main goals of developing such models is to '
  'improve their ability to understand and generate natural language text, '
  'particularly in more complex and nuanced scenarios. To test its '
  'capabilities in such scenarios, GPT-4 was evaluated on a variety of exams '
  'originally designed for humans.',
  14.397660909761523)]
"""


pprint("#####")
pprint(rag("what's the key findings of the paper?"))
"""
[('Below is part of the InstuctGPT paper. Could you read and summarize it to '
  'me?',
  18.415021371669887),
 ('What is the sum of average daily meat consumption for Georgia and Western '
  'Asia? Provide a step-by-step reasoning before providing your answer.',
  14.911014094212106),
 ('JWT Secret Hardcoded: The JWT secret key is hardcoded in the '
  "``loginHandler'' function, which is not a good practice. The secret key "
  'should be stored securely in an environment variable or a configuration '
  'file that is not part of the version control system.4.',
  14.233930972800035)]
"""

"""
What problems do you notice ...? 
"""