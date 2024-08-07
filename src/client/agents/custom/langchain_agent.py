from typing import List
from dotenv import load_dotenv

from src.client import AgentClient

_ = load_dotenv()
from langchain_openai import ChatOpenAI

def prompter(messages, role_key="role", content_key="content", user_role="user", agent_role="agent"):
    role_dict = {
        "user": user_role,
        "agent": agent_role,
    }
    prompt = []
    for item in messages:
        prompt.append(
            {role_key: role_dict[item["role"]], content_key: item["content"]}
        )
    return prompt

class LangChainAgent(AgentClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def inference(self, history: List[dict]) -> str:
        # return "I received {} items in history.".format(len(history))

        history = prompter(history, agent_role="assistant")

        message = self.model.invoke(history)
        
        return message.content
        

