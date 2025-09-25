from murf import Murf
from src.llm.ollama import Ollama
from src.chat.agent import build_journal_agent_executor
from langchain_community.chat_message_histories import RedisChatMessageHistory
import json


class ChatService:
    def __init__(self, session_id: str):
        self.llm = Ollama().llm
        self.memory = RedisChatMessageHistory(session_id=session_id, url="redis://localhost:6379")
        self.agent_executor = build_journal_agent_executor(self.llm)
        self.tts = Murf(api_key="ap2_0e47acd0-061f-436a-a069-9828312a5af9")

    def chat(self, description: str, input_message: str):
        try:
            self.memory.add_user_message(input_message)
            result = self.agent_executor.invoke({"desc": description, "history": self.memory.messages})
            return self.__parse_result(result["output"])
        except Exception as e:
            return {"type": "error", "content": str(e)}

    def get_chat_history(self):
        """
        Retrieve all chat messages for the current session from Redis.
        
        Returns:
            List of message dictionaries with type, content, and metadata
        """
        try:
            messages = self.memory.messages
            chat_history = []
            
            for message in messages:
                message_dict = {
                    "type": "user" if hasattr(message, 'type') and message.type == "human" else "ai",
                    "content": message.content,
                    "timestamp": getattr(message, 'timestamp', None)
                }
                chat_history.append(message_dict)
            
            return chat_history
        except Exception as e:
            raise Exception(f"Failed to retrieve chat history: {str(e)}")

    def __parse_result(self, result: str):
        if result.find("QUESTION:") != -1:
            result = result[result.find("QUESTION:") + len("QUESTION:") :].strip()
            response_type = "question"
        elif result.find("JOURNAL:") != -1:
            result = result[result.find("JOURNAL:") + len("JOURNAL:") :].strip()
            response_type = "journal"
        else:
            result = result if len(result.split(":")) < 1 else result.split(":")[1].strip()
            response_type = "response"
        audio_file = self.tts.text_to_speech.generate(
            text=result,
            voice_id="en-US-natalie",
        ).audio_file
        return {"type": response_type, "content": result, "audio_file": audio_file}
