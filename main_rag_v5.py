from tinyrag.rag_v5 import RAGVer5
from dotenv import load_dotenv
load_dotenv()
rag = RAGVer5()

answer = rag("What was the main objective of the paper?", alpha=0.6)
print(answer)
"""
I'm afraid I can't answer your question due to insufficient evidence.
Here is the reason:  The excerpts provided do not provide enough information to answer the user query.
        Final Answer: No.
"""


print("######")
answer = rag("When was the paper published?", alpha=0.6)  
print(answer)
"""
I'm afraid I can't answer your question due to insufficient evidence.
Here is the reason:  The excerpts provided do not mention the date of publication of the paper.
        Final Answer: No.
"""


print("######")
answer = rag("How did the authors tested GPT4?", alpha=0.6)
print(answer)
"""
The authors tested GPT-4 by evaluating its performance on a diverse set of benchmarks, including exams that were originally designed for humans [1]. The authors did not specifically train GPT-4 for these exams. Some of the problems in the exams were seen by the model during training, but for each exam, they also ran a variant with these questions removed and reported the lower score of the two [1]. GPT-4 performed well on these academic benchmarks [3] and often outscored the vast majority of human test takers [2].
--- EXCERPTS ---
[1]. "We tested GPT-4 on a diverse set of benchmarks, including simulating exams that were originally designed for humans.4 We did no speciﬁc training for these exams. A minority of the problems in the exams were seen by the model during training; for each exam we run a variant with these questions removed and report the lower score of the two."
[2]. "To test its capabilities in such scenarios, GPT-4 was evaluated on a variety of exams originally designed for humans. In these evaluations it performs quite well and often outscores the vast majority of human test takers."
[3]. "Table 2. Performance of GPT-4 on academic benchmarks."
"""

print("######")
answer = rag("In what ways is GPT4 limited by?", alpha=0.6)
print(answer)
"""
GPT-4 is limited in several ways, as mentioned in the paper "GPT-4 Technical Report" [1][2][3]. Despite its capabilities, GPT-4 still suffers from limitations similar to earlier GPT models. One major limitation is its lack of full reliability, as it may "hallucinate" facts and make reasoning errors [1]. Additionally, GPT-4 has a limited context window, which restricts its understanding and processing of larger bodies of text [2]. These limitations pose significant and novel safety challenges, highlighting the need for extensive research in areas like bias, disinformation, over-reliance, privacy, cybersecurity, and proliferation [3].
--- EXCERPTS ---
[1]. "Despite its capabilities, GPT-4 has similar limitations as earlier GPT models. Most importantly, it still is not fully reliable (it “hallucinates” facts and makes reasoning errors)."
[2]. "Despite its capabilities, GPT-4 has similar limitations to earlier GPT models [1, 37, 38]: it is not fully reliable (e.g. can suffer from “hallucinations”), has a limited context window, and does not learn∗Please cite this work as “OpenAI (2023)". Full authorship contribution statements appear at the end of thedocument."
[3]. "GPT-4’s capabilities and limitations create signiﬁcant and novel safety challenges, and we believe careful study of these challenges is an important area of research given the potential societal impact. This report includes an extensive system card (after the Appendix) describing some of the risks we foresee around bias, disinformation, over-reliance, privacy, cybersecurity, proliferation, and more."
"""


print("######")
answer = rag("Does GPT4 demonstrate near-human intelligence?", alpha=0.6)
print(answer)
"""
Based on the given excerpts from the paper "GPT-4 Technical Report," there is evidence that GPT-4 demonstrates near-human intelligence. 

Excerpt [1] states that GPT-4 was evaluated on exams designed for humans and performs quite well, often outscoring the majority of human test takers. This suggests that GPT-4 exhibits a level of intelligence that is comparable to or even surpasses humans in certain scenarios.

Excerpt [2] further supports this, stating that GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score among the top 10% of test takers. This indicates that in these specific domains, GPT-4 can perform at a level comparable to that of human experts.

It should be noted, however, that both excerpts [2] and [3] also mention that GPT-4 is "less capable than humans in many real-world scenarios." This suggests that while GPT-4 may demonstrate near-human intelligence in specific domains, it may not possess the same level of general intelligence or adaptability as humans.
--- EXCERPTS ---
[1]. "To test its capabilities in such scenarios, GPT-4 was evaluated on a variety of exams originally designed for humans. In these evaluations it performs quite well and often outscores the vast majority of human test takers."
[2]. "While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score around the top 10% of test takers. GPT-4 is a Transformer- based model pre-trained to predict the next token in a document."
[3]. "We report the development of GPT-4, a large-scale, multimodal model which can accept image and text inputs and produce text outputs. While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score around the top 10% of test takers."
"""