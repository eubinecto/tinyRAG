

class RagVer5: 
    """
    improving the reader 📖:
    Fact-checking with simple prompt (the approach used by nemoguardrails)
    references:
    - Natural Language Inference (Papers with Code, 2022) - https://paperswithcode.com/task/natural-language-inference/latest
    - example fact-checking prompt from NemoGuardrails (Nvidia, 2023) - https://github.com/NVIDIA/NeMo-Guardrails/blob/44f637e6bf8484e1c0109a3ff0f9a54844850267/nemoguardrails/llm/prompts/general.yml#L92-L94
    - fact checking with prompt chaining (an indie hacker from Github): https://github.com/jagilley/fact-checker
    """
    
    def __init__(self):
        # build dtm, upsert vectors, etc.
        pass
    
    def __call__(self, query: str) -> str:
        pass