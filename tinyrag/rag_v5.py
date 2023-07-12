from .rag_v3 import RAGVer3
import openai
import guidance


class RAGVer5(RAGVer3): 
    """
    improving the reader 📖:
    fact-check before you answer. 
    using Chain-of-Thought with guidance (the approach used by nemoguardrails)
    references:
    - CoT:
    - guidance: 
    """ 

    def __call__(self, query: str, alpha: float = 0.4, k: int = 3) -> str:
        results: list[tuple[str, float]] = super().__call__(query, alpha, k)
        excerpts = [res[0] for res in results]
        excerpts = '\n'.join([f'[{i}]. \"{excerpt}\"' for i, excerpt in enumerate(excerpts, start=1)])
        # first, check if the query is answerable 
        guidance.llm = guidance.llms.OpenAI("text-davinci-003")
        # define the guidance program
        moderation_program = guidance(
        """
        title of the paper:
        {{title}}
        
        excerpts: 
        {{excerpts}}
        ---
        Given the excerpts from the paper, judge whether you can answer a user query or not.

        Answer in the following format:
        Query: the question asked by the user 
        Reasoning: show your reasoning on whether or not you can answer the user query with the excerpts provided
        Final Answer: either conclude with Yes or No
        ---
        {{~! place the real question at the end }}
        Query: {{query}}
        Reasoning: {{gen "reasoning" stop="\\n"}}
        Final Answer:{{#select "answer"}}Yes{{or}}No{{/select}}""")
        out = moderation_program(
            query=query,
            title=self.title,
            excerpts=excerpts
        )
        answer = out['answer'].strip()
        if answer == 'No':
            answer = "I'm afraid I can't answer your question due to insufficient evidence."
            answer += f"(Reason: {out['reasoning'].split('.')[0]}"
            return answer
        else:
            # proceed to answer
            prompt = f"""
            user query:
            {query}
            
            title of the paper:
            {self.title}
            
            excerpts: 
            {excerpts}
            ---
            given the excerpts from the paper above, answer the user query.
            In your answer, make sure to cite the excerpts by its number wherever appropriate.
            Note, however, that the excerpts may not be relevant to the user query.
            """
            chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                        messages=[{"role": "user", "content": prompt}])
            answer = chat_completion.choices[0].message.content
            answer += f"\n--- EXCERPTS ---\n{excerpts}"
            return answer