import os
from pathlib import Path

from pydantic_settings import BaseSettings
from typing import List

# .env 固定位于项目根目录（app/core/config.py 上两级），不依赖启动时的工作目录
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    """应用配置，从 .env 读取"""

    # 服务
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # LLM
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_MODEL: str = "qwen-plus"

    # 天气
    WEATHER_API_KEY: str = ""

    # 文件白名单
    FILE_WHITELIST: str = ""

    # TTS
    TTS_PROVIDER: str = "edge"
    EDGE_TTS_VOICE: str = "zh-CN-XiaoxiaoNeural"

    # ASR 语音识别
    ASR_PROVIDER: str = ""  # dashscope / baidu，空=禁用
    ASR_BASE_URL: str = ""  # 留空则从 LLM_BASE_URL 推导（同 host + /audio/asr）
    ASR_MODEL: str = "paraformer-v2"

    # 百度 TTS（预留）
    BAIDU_TTS_APP_ID: str = ""
    BAIDU_TTS_API_KEY: str = ""
    BAIDU_TTS_SECRET_KEY: str = ""

    # LangSmith 追踪
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_TRACING: str = "false"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_PROJECT: str = "AI-tablepet"

    # ===== MCP 配置 =====
    # 是否启用 MCP 工具（文件操作 + 网页获取），连接失败自动回退本地工具
    MCP_ENABLED: bool = True
    # filesystem MCP 允许访问的目录（分号分隔），为空则不启用文件 MCP
    MCP_FILESYSTEM_ROOTS: str = ""

    # RAG 向量知识库（Milvus + 阿里云 Embedding）
    VECTOR_URL: str = "http://localhost:19530"
    VECTOR_DIM: int = 1024
    EMBEDDING_MODEL: str = "text-embedding-v3"  # DashScope 1024 维
    # 留空跟随 LLM_BASE_URL；私有端点无 embedding 模型时改公共端点：
    # https://dashscope.aliyuncs.com/compatible-mode/v1
    EMBEDDING_BASE_URL: str = ""

    # PostgreSQL 持久化记忆（LangGraph checkpointer）
    POSTGRES_URI: str = "postgresql://postgres:123456@localhost:5432/langgraph"

    @property
    def mcp_filesystem_roots(self) -> List[str]:
        """filesystem MCP 允许访问的目录列表"""
        if not self.MCP_FILESYSTEM_ROOTS:
            return []
        return [p.strip() for p in self.MCP_FILESYSTEM_ROOTS.split(";") if p.strip()]

    @property
    def file_whitelist(self) -> List[str]:
        """文件白名单目录列表"""
        if not self.FILE_WHITELIST:
            return []
        return [p.strip() for p in self.FILE_WHITELIST.split(";") if p.strip()]

    def sync_langsmith_env(self) -> None:
        """把 LangSmith 配置写入进程环境变量。

        LangChain/LangGraph 的运行时追踪是通过读取 os.environ 开启的，
        pydantic-settings 只是把值读进本对象，必须回写环境变量才能生效。
        """
        mapping = {
            "LANGSMITH_API_KEY": self.LANGSMITH_API_KEY,
            "LANGSMITH_TRACING": self.LANGSMITH_TRACING,
            "LANGSMITH_ENDPOINT": self.LANGSMITH_ENDPOINT,
            "LANGSMITH_PROJECT": self.LANGSMITH_PROJECT,
        }
        for key, value in mapping.items():
            if value:
                os.environ[key] = value

    class Config:
        env_file = str(_ENV_FILE)
        env_file_encoding = "utf-8"


settings = Settings()
# 在任何 LLM/Agent 被调用前同步，LangSmith 即可完整追踪 LangGraph 图的每一步
settings.sync_langsmith_env()

