from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.core.config import settings


class LLMService:
    """大语言模型调用服务，基于 LangChain"""

    def __init__(self):
        self._llm: Optional[ChatOpenAI] = None

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                api_key=settings.LLM_API_KEY,
                base_url=settings.LLM_BASE_URL,
                model=settings.LLM_MODEL,
                temperature=0.7,
            )
        return self._llm

    async def chat(self, message: str, system_prompt: Optional[str] = None) -> str:
        """
        与 LLM 对话
        :param message: 用户消息
        :param system_prompt: 系统提示词，默认为桌宠角色
        :return: AI 回复
        """
        if system_prompt is None:
            system_prompt = (
                "你是一个可爱的AI桌宠，名字叫小桌。"
                "你性格活泼、友善，喜欢用轻松幽默的语气和用户交流。"
                "回复要简洁，适合桌面宠物的对话场景。"
            )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message),
        ]
        response = await self.llm.ainvoke(messages)
        return response.content


llm_service = LLMService()
