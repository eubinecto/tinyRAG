from rag101.rag_v3 import RAGVer3
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()


rag = RAGVer3()


# searching for a keyword
pprint(rag("the main goal"))

"""
[('ada, babbage, and curie refer to models available via the OpenAI API '
  '[47].We believe that accurately predicting future capabilities is important '
  'for safety. Going forward we plan to reﬁne these methods and register '
  'performance predictions across various capabilities before large model '
  'training begins, and we hope this becomes a common goal in the ﬁeld.',
  '0.009836066'),
 ('Predictions on the other ﬁve buckets performed almost as well, the main '
  'exception being GPT-4 underperforming our predictions on the easiest '
  'bucket. Certain capabilities remain hard to predict.',
  '0.009677419'),
 ('In contrast, the other options listed do not seem to be directly related to '
  'the title or themes of the work. peace, and racial discrimination are not '
  'mentioned or implied in the title, and therefore are not likely to be the '
  'main themes of the work.',
  '0.00952381')]
"""

# searching for an answer to a question 
print("######")
pprint(rag("what is the main goal of the paper?"))
"""
[('Below is part of the InstuctGPT paper. Could you read and summarize it to '
  'me?',
  '0.016393442'),
 ('What is funny about this image? Describe it panel by panel.', '0.009677419'),
 ('ada, babbage, and curie refer to models available via the OpenAI API '
  '[47].We believe that accurately predicting future capabilities is important '
  'for safety. Going forward we plan to reﬁne these methods and register '
  'performance predictions across various capabilities before large model '
  'training begins, and we hope this becomes a common goal in the ﬁeld.',
  '0.00952381')]
"""

# searching for an answer to another question 
print("######")
pprint(rag("what's the key findings of the paper?"))

"""
[('The InstructGPT paper focuses on training large language models to follow '
  'instructions with human feedback. The authors note that making language '
  'models larger doesn’t inherently make them better at following a user’s '
  'intent.',
  '0.009836066'),
 ("SignedString(key)'' function, which could lead to unexpected behavior.",
  '0.009677419'),
 ('JWT Secret Hardcoded: The JWT secret key is hardcoded in the '
  "``loginHandler'' function, which is not a good practice. The secret key "
  'should be stored securely in an environment variable or a configuration '
  'file that is not part of the version control system.4.',
  '0.00952381')]
"""

"""
What problems do you notice ...? 
"""