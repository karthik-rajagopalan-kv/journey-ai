import uuid
from typing import Any, List, Union
from langchain.prompts import PromptTemplate
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.agents import AgentExecutor, BaseSingleActionAgent
from langchain.schema import AgentAction, AgentFinish
from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseLLM
from langchain_core.messages import BaseMessage, SystemMessage
from redis import Redis

from src.llm.ollama import Ollama


class JournalAgent(BaseSingleActionAgent):
    """
    An agent that, given image + history + answers so far, decides whether to ask another question
    or generate the journal.
    """

    llm: BaseLLM

    def __init__(self, llm: BaseLLM):
        super().__init__(llm=llm)

    @property
    def input_keys(self):
        # What data the agent expects each time it is called
        return ["desc", "history"]  # history is full conversation history

    def plan(self, intermediate_steps: list[tuple[AgentAction, str]], callbacks: Callbacks = None, **kwargs):
        """
        Given inputs, return either a question to ask next, or finish with journal.
        """

        desc = kwargs["desc"]
        history = kwargs["history"]  # chat history (system / user / agent messages)

        prompt_template = PromptTemplate(
            input_variables=["desc"],
            template=(
                "You are a journal writing assistant. Here is a small synopsis of an event based on the user. It can be description of an image, a video, or a story:\n"
                "{desc}\n\n"
                "## INSTRUCTIONS: \n"
                "- Ask a question to the user based on the small synopsis of the event to the user to write a journal entry. "
                "ALWAYS ask questions one by one.\n"
                "Prefix the question with `QUESTION:`.\n"
                "- If you have enough information from the user, write a journal entry based on the synopsis and the answers given by the user. "
                "Prefix the journal entry with `JOURNAL:`.\n"
                "- If the user has any improvements to the journal entry, include them in the journal entry. "
                "It should be prefixed with `FOLLOWUP:`.\n"
                "\n"
                "## GUIDELINES:\n"
                "- ALWAYS ask friendly, engaging, and context-aware questions based on the description.\n"
                "- ALWAYS prefix with exactly one of: `QUESTION:`\n"
                "- DO NOT use any other prefixes, labels, or extra text.\n"
                "- DO NOT include dates in the journal entry.\n"
                "- ALWAYS write the journal in the first-person perspective.\n"
                "- ALWAYS ask atleast 2 questions.\n"
                "- NEVER repeat questions.\n"
            ),
        )

        prompt = prompt_template.format(desc=desc)
        messages: List[BaseMessage] = [SystemMessage(content=prompt)] + history
        llm_output = self.llm.invoke(messages)
        return AgentFinish(return_values={"output": llm_output.strip()}, log=llm_output.strip())

    async def aplan(
        self, intermediate_steps: list[tuple[AgentAction, str]], callbacks: Callbacks = None, **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        return await super().aplan(intermediate_steps, callbacks, **kwargs)


def build_journal_agent_executor(llm):
    agent = JournalAgent(llm=llm)
    tools = []  # if you want to call describe tool as needed

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
    )
    return agent_executor


def run(image_url: str):
    from src.descriptors.descriptor import Descriptor

    descriptor = Descriptor()
    llm: BaseLLM = Ollama().llm
    memory = RedisChatMessageHistory(session_id=str(uuid.uuid4()), url="redis://localhost:6379")
    agent_exec = build_journal_agent_executor(llm)

    print("Agent executor built")

    desc = descriptor.describe_image(image_url)
    print("Desc:", desc)
    qa_history = []
    journal = None

    input_message = "Hi, help me write a journal."
    while True:
        # Call agent with current state
        memory.add_user_message(input_message)
        result = agent_exec.invoke({"desc": desc, "history": memory.messages})  # can be empty; user input comes later

        if result["output"].find("QUESTION:") != -1:
            question = result["output"][result["output"].find("QUESTION:") + len("QUESTION:") :].strip()
            print("Q:", question)
            input_message = input("> ")
            qa_history.append((question, input_message))
            memory.add_ai_message(question)
        elif result["output"].find("FOLLOWUP:") != -1:
            journal = result["output"][result["output"].find("FOLLOWUP:") + len("FOLLOWUP:") :].strip()
            memory.add_ai_message(journal)
        elif result["output"].find("JOURNAL:") != -1:
            journal = result["output"][result["output"].find("JOURNAL:") + len("JOURNAL:") :].strip()
            memory.add_ai_message(journal)
            confirm = input("Do you want to continue? (y/n)")
            print("Confirm:", confirm)
            if confirm.strip() == "n":
                break
            else:
                break
    print("---------------------------------------------------")
    print("Journal:", journal)
    print("---------------------------------------------------")


class Summarizer:
    def __init__(self) -> None:
        self.llm = Ollama().llm
        self.redis_client = Redis()

    def summarize_sessions(self, session_ids: list[str]):
        try:
            system_prompt = """Based on different small journal entries. Write an elaborate journal.\n
            Journal entries: {journal_entries} \n\n
            ALWAYS return only the journal entry, no other text.
            """
            journal_entries = []
            for session_id in session_ids:
                journal = self.redis_client.get(f"journal:{session_id}")
                if journal is None:
                    print(f"Journal not found for session id: {session_id}")
                    continue
                journal_entries.append(journal)
            prompt = system_prompt.format(
                journal_entries="\n".join([f"{index+1}. {journal}" for index, journal in enumerate(journal_entries)])
            )
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            print(f"Error summarizing sessions: {e}")
            return ""
