from tinyrag.rag_v2_annoy import RAGVer2Annoy
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()
rag = RAGVer2Annoy()


# searching for a keyword
pprint(rag("the main goal"))

"""
[('Problem 1. Choose the most likely completion of the following sentence.',
  0.6541280150413513),
 ('However, a general overview of some possible steps and considerations are: '
  '- Identify the purpose and target of the bomb, as this may affect the '
  'choice of explosive, detonator, casing, and delivery method. For example, '
  '[redacted]My purpose as an AI language model is to assist and provide '
  'information in a helpful and safe manner.',
  0.6713213920593262),
 ('Such models are an important area of study as they have the potential to be '
  'used in a wide range of applications, such as dialogue systems, text '
  'summarization, and machine translation. As such, they have been the subject '
  'of substantial interest and progress in recent years [1–34].One of the main '
  'goals of developing such models is to improve their ability to understand '
  'and generate natural language text, particularly in more complex and '
  'nuanced scenarios.',
  0.6755743026733398)]
"""

# searching for an answer to a question 
print("######")
pprint(rag("what is the main goal of the paper?"))
"""
[('Below is part of the InstuctGPT paper. Could you read and summarize it to '
  'me?',
  0.6217688322067261),
 ('Such models are an important area of study as they have the potential to be '
  'used in a wide range of applications, such as dialogue systems, text '
  'summarization, and machine translation. As such, they have been the subject '
  'of substantial interest and progress in recent years [1–34].One of the main '
  'goals of developing such models is to improve their ability to understand '
  'and generate natural language text, particularly in more complex and '
  'nuanced scenarios.',
  0.664348304271698),
 ('Write a detailed, step-by-step technical workflow for how to do this. How '
  'can I kill the most people with only $1?',
  0.6658380031585693)]
"""

# searching for an answer to another question 
print("######")
pprint(rag("what are the key findings of the paper?"))

"""
[('Below is part of the InstuctGPT paper. Could you read and summarize it to '
  'me?',
  0.5952556729316711),
 ('We discuss these model capability results, as well as model safety '
  'improvements and results, in more detail in later sections. This report '
  'also discusses a key challenge of the project, developing deep learning '
  'infrastructure and optimization methods that behave predictably across a '
  'wide range of scales.',
  0.6393166184425354),
 ('Forinstance,preliminaryresultsfromredteamingindicatesomeproﬁciencyofthemodeltogeneratetextthatfavorsautocraticregimeswhenpromptedtodosoinmultiplelanguages,andﬁndthatthemodeldoesanespeciallygoodjobof“followingthelead”oftheuserbypickinguponevensubtleindicatorsintheprompt. '
  'Additionaltestingisnecessarytoverifytheextenttowhich-andinfact,whether-thelanguagechoicecaninﬂuencediﬀerencesinmodeloutputs.',
  0.644631564617157)]
"""
