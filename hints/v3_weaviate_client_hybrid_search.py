import weaviate
from dotenv import load_dotenv
import os
load_dotenv()

credentials = weaviate.auth.AuthApiKey(os.environ['WEAVIATE_CLUSTER_KEY'])
client = weaviate.Client(
   os.environ['WEAVIATE_CLUSTER_URL'],
   credentials,
   additional_headers={
      'X-OpenAI-Api-Key': os.environ['OPENAI_API_KEY']
   }
   )
print(client.is_ready())
"""
True
"""

# ---first, flush the cluster --- #
client.schema.delete_all()


# --- create a schema --- #
class_obj = {
    "class": "Sentence",
    "moduleConfig": {
        "text2vec-openai": {
            "vectorizeClassName": False,
            "model": "ada",
            "modelVersion": "002",  #  we are using ada
            "type": "text"
        }
    },
    "properties": [
        {
            "name": "content",
            "dataType": ["text"],
        }
    ],
    "vectorizer": "text2vec-openai"
}

print(client.schema.create_class(class_obj))


sentences = ['As such, they have been the subject of substantial interest and progress in '
  'recent years [1–34].One of the main goals of developing such models is to '
  'improve their ability to understand and generate natural language text, '
  'particularly in more complex and nuanced scenarios. To test its '
  'capabilities in such scenarios, GPT-4 was evaluated on a variety of exams '
  'originally designed for humans.',
 'Such models are an important area of study as they have the potential to be '
  'used in a wide range of applications, such as dialogue systems, text '
  'summarization, and machine translation. As such, they have been the subject '
  'of substantial interest and progress in recent years [1–34].One of the main '
  'goals of developing such models is to improve their ability to understand '
  'and generate natural language text, particularly in more complex and '
  'nuanced scenarios.',
  'Predictions on the other ﬁve buckets performed almost as well, the main '
  'exception being GPT-4 underperforming our predictions on the easiest '
  'bucket. Certain capabilities remain hard to predict.',
  ]


# --- upload corpus as a batch --- #
def check_batch_result(results: dict):
  """
  Check batch results for errors.

  Parameters
  ----------
  results : dict
      The Weaviate batch creation return value.
  """

  if results is not None:
    for result in results:
      if "result" in result and "errors" in result["result"]:
        if "error" in result["result"]["errors"]:
          print(result["result"])

with client.batch(
    batch_size=3,               # Specify batch size
    num_workers=4,             # Parallelize the process
    dynamic=True,                        # Enable/Disable dynamic batch size change
    timeout_retries=3,           # Number of retries if a timeout occurs
    connection_error_retries=3,  # Number of retries if a connection error occurs
    callback=check_batch_result,
) as batch:
    for sent in sentences:
        batch.add_data_object(
            {'content': sent},
            class_name="Sentence"
        )


#  --- retrieve vectors --- #
r = client.data_object.get(
   with_vector=True, 
   class_name="Sentence"
)
for obj in r['objects']: 
   print(len(obj['vector']))
   print("first 10 dimensions:")
   print(obj['vector'][:10])
   print("----")

"""
1536
first 10 dimensions:
[-0.03189032, 0.0018862827, 0.020012407, -0.018030595, 0.00643441, 0.014701671, -0.012648618, 0.01895026, 4.3842916e-05, -0.044687897]
----
1536
first 10 dimensions:
[-0.0064870357, -0.024078006, 0.012719678, -0.030142197, 0.013070328, 0.009330056, -0.009921349, 0.0117158545, -0.03454252, -0.008092465]
----
1536
first 10 dimensions:
[-0.018601794, -0.004305233, 0.020035764, -0.021682175, 0.014618539, 0.010290071, 0.0039766147, 0.016942104, -0.017327152, -0.02939641]
----
"""


# --- search over vectors semantically --- #
r = (
    client.query
    .get("Sentence", ["content"])
    # it is just this part that you need fixing
    .with_hybrid(
        query="what is the main goal of the paper?"
    )
    .with_additional(["score", "explainScore"])
    .with_limit(3)
    .do()
)

from pprint import pprint
pprint(r)

"""
{'data': {'Get': {'Sentence': [{'_additional': {'explainScore': '(bm25)\n'
                                                                '(hybrid) '
                                                                'Document '
                                                                '77ccec80-786c-43b8-a5cc-43671af3c3e6 '
                                                                'contributed '
                                                                '0.003968253968253968 '
                                                                'to the score\n'
                                                                '(hybrid) '
                                                                'Document '
                                                                '77ccec80-786c-43b8-a5cc-43671af3c3e6 '
                                                                'contributed '
                                                                '0.012295081967213115 '
                                                                'to the score',
                                                'score': '0.016263336'},
                                'content': 'Such models are an important area '
                                           'of study as they have the '
                                           'potential to be used in a wide '
                                           'range of applications, such as '
                                           'dialogue systems, text '
                                           'summarization, and machine '
                                           'translation. As such, they have '
                                           'been the subject of substantial '
                                           'interest and progress in recent '
                                           'years [1–34].One of the main goals '
                                           'of developing such models is to '
                                           'improve their ability to '
                                           'understand and generate natural '
                                           'language text, particularly in '
                                           'more complex and nuanced '
                                           'scenarios.'},
                               {'_additional': {'explainScore': '(bm25)\n'
                                                                '(hybrid) '
                                                                'Document '
                                                                '08f8880a-443d-4233-8f55-90fa6b1b8637 '
                                                                'contributed '
                                                                '0.004032258064516129 '
                                                                'to the score\n'
                                                                '(hybrid) '
                                                                'Document '
                                                                '08f8880a-443d-4233-8f55-90fa6b1b8637 '
                                                                'contributed '
                                                                '0.012096774193548387 '
                                                                'to the score',
                                                'score': '0.016129032'},
                                'content': 'As such, they have been the '
                                           'subject of substantial interest '
                                           'and progress in recent years '
                                           '[1–34].One of the main goals of '
                                           'developing such models is to '
                                           'improve their ability to '
                                           'understand and generate natural '
                                           'language text, particularly in '
                                           'more complex and nuanced '
                                           'scenarios. To test its '
                                           'capabilities in such scenarios, '
                                           'GPT-4 was evaluated on a variety '
                                           'of exams originally designed for '
                                           'humans.'},
                               {'_additional': {'explainScore': '(bm25)\n'
                                                                '(hybrid) '
                                                                'Document '
                                                                '9ea82793-8cd5-4764-a894-67aab44a1205 '
                                                                'contributed '
                                                                '0.004098360655737705 '
                                                                'to the score\n'
                                                                '(hybrid) '
                                                                'Document '
                                                                '9ea82793-8cd5-4764-a894-67aab44a1205 '
                                                                'contributed '
                                                                '0.011904761904761904 '
                                                                'to the score',
                                                'score': '0.016003123'},
                                'content': 'Predictions on the other ﬁve '
                                           'buckets performed almost as well, '
                                           'the main exception being GPT-4 '
                                           'underperforming our predictions on '
                                           'the easiest bucket. Certain '
                                           'capabilities remain hard to '
                                           'predict.'}]}}}
"""

