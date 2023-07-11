import spacy
from spacy.lang.en import English

text = 'My first birthday was great. My 2? Even better.'
nlp_simple = English()
nlp_simple.add_pipe('sentencizer')  # rule-based


for sent in nlp_simple(text).sents:
    print(sent.text)

"""
My first birthday was great.
My 2?
Even better.
"""

nlp_better = spacy.load('en_core_web_sm')  # # context-aware 


print("#######")
for sent in nlp_better(text).sents:
    print(sent.text)


"""
My first birthday was great.
My 2? Even better.
"""
