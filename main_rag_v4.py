from tinyrag.rag_v4 import RAGVer4
from dotenv import load_dotenv
load_dotenv()
rag = RAGVer4()


answer = rag("What was the main objective of the paper?", alpha=0.6)
print(answer)
"""
The excerpts provided do not mention the main objective of the paper.
--- EXCERPTS ---
[1]. Below is part of the InstuctGPT paper. Could you read and summarize it to me?
[2]. Problem 1. Choose the most likely completion of the following sentence.
[3]. We cross-checked these materials against the model’s training data to determine the extent to which the training data was not contaminated with any exam questions, which we also report in this paper. The Uniform Bar Exam was run by our collaborators at CaseText and Stanford CodeX.
"""


print("######")
answer = rag("When was the paper published?", alpha=0.6)
print(answer)

"""
The specific publication date of the paper "GPT-4 Technical Report" is not mentioned in the provided excerpts. Thus, the information regarding the publication date of the paper is not available.
--- EXCERPTS ---
[1]. Below is part of the InstuctGPT paper. Could you read and summarize it to me?
[2]. We sourced either the most recent publicly-available ofﬁcial past exams, or practice exams in published third-party 2022-2023 study material which we purchased. We cross-checked these materials against the model’s training data to determine the extent to which the training data was not contaminated with any exam questions, which we also report in this paper.
[3]. arXiv preprint arXiv:2205.11916, 2022.ing sentiment. arXiv preprint arXiv:1704.01444, 2017.
"""

print("######")
answer = rag("How did the authors tested GPT4?", alpha=0.6)
print(answer)

"""
The authors of the paper tested GPT-4 on a diverse set of benchmarks, including simulating exams that were originally designed for humans [1]. They did not provide specific training for these exams. Some of the problems in the exams were seen by the model during training, but for each exam, they removed these questions and reported the lower score of the two [1]. GPT-4 was evaluated on a variety of exams originally designed for humans to test its capabilities in these scenarios. It performed well in these evaluations and often outscored the majority of human test takers [2]. The performance of GPT-4 on academic benchmarks is presented in Table 2 [3].
--- EXCERPTS ---
[1]. We tested GPT-4 on a diverse set of benchmarks, including simulating exams that were originally designed for humans.4 We did no speciﬁc training for these exams. A minority of the problems in the exams were seen by the model during training; for each exam we run a variant with these questions removed and report the lower score of the two.
[2]. To test its capabilities in such scenarios, GPT-4 was evaluated on a variety of exams originally designed for humans. In these evaluations it performs quite well and often outscores the vast majority of human test takers.
[3]. Table 2. Performance of GPT-4 on academic benchmarks.
"""


print("######")
answer = rag("In what ways is GPT4 limited by?", alpha=0.6)
print(answer)

"""
GPT-4, despite its capabilities, is still limited in several ways. According to excerpt [1], GPT-4 shares similar limitations as earlier GPT models. It is not fully reliable and can "hallucinate" facts and make reasoning errors. This is reiterated in excerpt [2], where it is mentioned that GPT-4 is not fully reliable, has a limited context window, and does not learn. 

Furthermore, in excerpt [3], it is highlighted that the capabilities and limitations of GPT-4 pose significant safety challenges. The paper emphasizes the importance of studying these challenges in areas such as bias, disinformation, over-reliance, privacy, cybersecurity, and proliferation.

Overall, GPT-4 has limitations in terms of reliability, context window, and learning, which need to be addressed to ensure its effectiveness and mitigate potential safety challenges.
--- EXCERPTS ---
[1]. Despite its capabilities, GPT-4 has similar limitations as earlier GPT models. Most importantly, it still is not fully reliable (it “hallucinates” facts and makes reasoning errors).
[2]. Despite its capabilities, GPT-4 has similar limitations to earlier GPT models [1, 37, 38]: it is not fully reliable (e.g. can suffer from “hallucinations”), has a limited context window, and does not learn∗Please cite this work as “OpenAI (2023)". Full authorship contribution statements appear at the end of thedocument.
[3]. GPT-4’s capabilities and limitations create signiﬁcant and novel safety challenges, and we believe careful study of these challenges is an important area of research given the potential societal impact. This report includes an extensive system card (after the Appendix) describing some of the risks we foresee around bias, disinformation, over-reliance, privacy, cybersecurity, proliferation, and more.
"""



print("######")
answer = rag("Does GPT4 demonstrate near-human intelligence?", alpha=0.6)
print(answer)

"""
Based on the excerpts from the paper, GPT-4 demonstrates human-level performance on various professional and academic benchmarks, including exams designed for humans and a simulated bar exam. It often outperforms the majority of human test takers [1][2][3]. Therefore, it can be inferred that GPT-4 exhibits near-human intelligence.
--- EXCERPTS ---
[1]. To test its capabilities in such scenarios, GPT-4 was evaluated on a variety of exams originally designed for humans. In these evaluations it performs quite well and often outscores the vast majority of human test takers.
[2]. While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score around the top 10% of test takers. GPT-4 is a Transformer- based model pre-trained to predict the next token in a document.
[3]. We report the development of GPT-4, a large-scale, multimodal model which can accept image and text inputs and produce text outputs. While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score around the top 10% of test takers.
"""
