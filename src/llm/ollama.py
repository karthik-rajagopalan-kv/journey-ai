from langchain_ollama import OllamaLLM

class Ollama:
    def __init__(self, model: str = "gemma3:4b"):
        self.llm = OllamaLLM(model=model)

    def generate(self, prompt: str) -> str:
        return self.llm.invoke(prompt)

    def bind(self, images: list[str]):
        llm = self.llm.bind(images=images)
        self.llm = llm