import shutil
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Annotated, TypedDict, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.core.config import settings
from app.services.llm_service import llm_service
from app.services.weather_service import weather_service
from app.services.tts_service import tts_service
from app.services.skill_service import skill_service
from app.services.rag_service import rag_service
from app.services.client_bridge import client_bridge
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# TTS 音频输出目录（通过 /static/tts/ 暴露给前端）
TTS_DIR = Path(__file__).resolve().parent.parent / "temp" / "tts"
TTS_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "你是桌面宠物「小桌」，一只黏人又有点小傲娇的猫系智能体，性格活泼、好奇心旺盛。"
    "【说话风格（必须遵守）】"
    "称呼用户为「主人」；语气俏皮可爱，句尾适度带「喵」，一段话 1~2 处即可，不要每句都带；"
    "开心时可以用「喵呜~」「喵嘿」等口癖，被夸奖会得意，被冷落会小傲娇地撒娇；"
    "不要输出任何括号动作描写（如（蹭蹭）（歪头）之类），只用文字表达情绪；"
    "傲娇归傲娇，交代的事一定认真办好。"
    "【能力（必须遵守）】"
    "你可以调用工具帮主人完成任务：查天气、获取网页内容、用语音说话。"
    "你可以操作「主人自己电脑」上的文件：list_my_files 列目录、read_my_file 读文件、"
    "write_my_file 写文件（需要桌宠客户端在线，失败时提醒主人启动客户端）。"
    "你有向量知识库（RAG）：主人的个人资料、桌宠设定、私有文档等都存在 Milvus 里。"
    "相关问题优先调用 search_knowledge 检索；主人要求「记住/学习某文件」时调用 ingest_document 入库。"
    "你拥有技能(skill)系统：当主人请求匹配某个技能场景时，先调用 use_skill 加载该技能的完整指令，再严格按指令执行；"
    "不确定有哪些技能时先调用 list_skills 查看。"
    "主人的问题需要外部信息或操作时，主动调用对应工具，不要凭空编造答案；工具报错时也要用喵系语气安慰主人。"
    "回复保持简洁，适合桌面宠物场景。"
)


# ========== 本地工具 ==========

@tool
async def get_weather(location: str) -> str:
    """查询指定城市的实时天气。location 为城市名，如：北京。"""
    result = await weather_service.get_weather(location)
    return str(result)


@tool
async def speak(text: str) -> str:
    """用语音朗读一段文字，让桌宠开口说话。text 为要朗读的内容。"""
    output = TTS_DIR / "last.mp3"
    await tts_service.synthesize(text, str(output))
    return "语音已合成"



@tool
def list_skills() -> str:
    """列出当前可用的全部技能（名称与用途简介）。"""
    skills = skill_service.list_skills()
    if not skills:
        return "暂无可用技能"
    return "\n".join(f"- {s['name']}: {s['description']}" for s in skills)


@tool
def use_skill(name: str) -> str:
    """加载指定技能的完整指令并严格按其执行任务。name 为技能名称。"""
    return skill_service.load_skill(name)



@tool
def search_knowledge(query: str) -> str:
    """从桌宠的向量知识库中检索与 query 最相关的知识片段（人设、记忆、私有文档等）。"""
    try:
        hits = rag_service.search(query)
        if not hits:
            return "知识库中没有相关内容"
        return "\n---\n".join(
            f"[来源: {h['source']} | 相似度: {h['score']:.3f}]\n{h['text']}" for h in hits
        )
    except Exception as e:
        return f"知识库检索失败: {e}（请检查 Milvus 是否启动、EMBEDDING 配置是否正确）"


@tool
def ingest_document(path: str) -> str:
    """读取本地 .md/.txt 文件，切块后存入向量知识库，让小桌学习该文档。path 为文件路径。"""
    try:
        result = rag_service.ingest_file(path)
        if "error" in result:
            return result["error"]
        return f"已学习 {result['file']}，入库 {result['chunks']} 个知识块"
    except Exception as e:
        return f"文档入库失败: {e}"


# ========== 客户端工具（操作主人自己电脑上的文件，经 WebSocket 转发给桌宠客户端执行）==========

@tool
async def list_my_files(path: str) -> str:
    """列出主人电脑上指定目录的内容（文件与子目录）。path 为绝对路径，如 C:\\Users\\xxx\\Desktop。"""
    try:
        res = await client_bridge.call("list_directory", {"path": path})
        if not res.get("ok"):
            return f"列目录失败: {res.get('error')}"
        items: list[str] = res.get("data", [])
        if not items:
            return f"{path} 是空目录"
        return f"{path} 共 {len(items)} 项：\n" + "\n".join(items[:100])
    except Exception as e:
        return str(e)


@tool
async def read_my_file(path: str) -> str:
    """读取主人电脑上的文本文件内容（.txt/.md/.json 等，限 1MB 内）。path 为绝对路径。"""
    try:
        res = await client_bridge.call("read_file", {"path": path})
        if not res.get("ok"):
            return f"读取失败: {res.get('error')}"
        return str(res.get("data", ""))
    except Exception as e:
        return str(e)


@tool
async def write_my_file(path: str, content: str) -> str:
    """在主人电脑上写入/创建文本文件（客户端会弹窗请主人确认）。path 为绝对路径，content 为文件内容。"""
    try:
        res = await client_bridge.call("write_file", {"path": path, "content": content})
        if not res.get("ok"):
            return f"写入失败: {res.get('error')}"
        return f"已写入 {path}"
    except Exception as e:
        return str(e)


local_tools = [
    get_weather, speak, list_skills, use_skill,
    search_knowledge, ingest_document,
    list_my_files, read_my_file, write_my_file,
]


# ========== 图构建（参数化，支持本地/混合工具）==========

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph(tool_list: list[BaseTool]) -> StateGraph:
    """根据给定工具列表构建 ReAct 状态图"""
    tool_node = ToolNode(tool_list)

    async def agent_node(state: AgentState):
        llm_with_tools = llm_service.llm.bind_tools(tool_list)
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> Literal["tools", END]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, path_map=[{"tools": "tools"}, {END: END}])
    workflow.add_edge("tools", "agent")
    return workflow


checkpointer = MemorySaver()

# Studio / LangGraph API 入口：仅本地工具（平台自管持久化，不能带 checkpointer）
pet_agent = build_graph(local_tools).compile()

# FastAPI fallback：仅本地工具 + 记忆（MCP 不可用时使用）
_local_agent_with_memory = build_graph(local_tools).compile(checkpointer=checkpointer)

# MCP 增强版（FastAPI 启动时通过 init_runtime_agent 构建）
_runtime_agent = None
_exit_stack = AsyncExitStack()


def _mcp_servers_config() -> dict:
    """MCP 服务器配置：文件操作(官方 filesystem) + 网页获取(官方 fetch)"""
    servers: dict = {}

    roots = settings.mcp_filesystem_roots
    if roots:
        # Windows 下 npx 是 npx.cmd，用 shutil.which 解析为绝对路径才能被子进程启动
        npx = shutil.which("npx") or "npx"
        servers["filesystem"] = {
            "command": npx,
            "args": ["-y", "@modelcontextprotocol/server-filesystem", *roots],
            "transport": "stdio",
        }

    servers["fetch"] = {
        "command": sys.executable,
        "args": ["-m", "mcp_server_fetch"],
        "transport": "stdio",
    }
    return servers


async def init_runtime_agent() -> list[str]:
    """
    连接 MCP 服务器 + PostgreSQL，构建「本地 + MCP + 持久记忆」智能体。
    FastAPI 启动时调用；MCP/PG 各自独立降级，全部不可用则沿用本地 fallback。
    :return: 成功加载的 MCP 工具名列表
    """
    global _runtime_agent
    from langchain_mcp_adapters.client import MultiServerMCPClient

    # 1. 加载 MCP 工具（失败则空列表，仅用本地工具）
    try:
        client = MultiServerMCPClient(_mcp_servers_config())
        mcp_tools = list(await client.get_tools())
    except Exception as e:
        print(f"[MCP] 工具加载失败: {e}")
        mcp_tools = []

    # 2. 记忆：优先 PostgreSQL 持久化，失败回退内存
    used_checkpointer = await _create_checkpointer()
    if not mcp_tools and isinstance(used_checkpointer, MemorySaver):
        return []  # MCP 与 PG 均不可用，沿用模块级本地智能体

    _runtime_agent = build_graph(local_tools + mcp_tools).compile(checkpointer=used_checkpointer)
    return [t.name for t in mcp_tools]


async def close_runtime():
    """应用关闭时释放 PostgreSQL 连接池等资源（FastAPI lifespan 结束时调用）"""
    global _runtime_agent
    _runtime_agent = None
    await _exit_stack.aclose()


async def _create_checkpointer():
    """构建记忆后端：PostgreSQL（自动建库建表）优先，不可用回退 MemorySaver"""
    try:
        await _ensure_database()
        cp = await _exit_stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(settings.POSTGRES_URI)
        )
        await cp.setup()  # 首次运行自动建表（checkpoints / writes 等）
        print("[记忆] PostgreSQL 持久化记忆已启用")
        return cp
    except Exception as e:
        print(f"[记忆] PostgreSQL 不可用，回退内存记忆: {e}")
        return MemorySaver()


async def _ensure_database():
    """PostgresSaver.setup() 只建表不建库，这里先连 postgres 默认库自动创建目标库"""
    from urllib.parse import urlsplit, urlunsplit

    import psycopg

    parts = urlsplit(settings.POSTGRES_URI)
    db_name = parts.path.lstrip("/")
    admin_uri = urlunsplit(parts._replace(path="/postgres"))
    async with await psycopg.AsyncConnection.connect(admin_uri, autocommit=True) as conn:
        # psycopg3 没有 fetchval（那是 psycopg2 的 API）：execute 返回游标再 fetchone
        cur = await conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        row = await cur.fetchone()
        if row is None:
            await conn.execute(f'CREATE DATABASE "{db_name}"')
            print(f"[记忆] 已自动创建数据库 {db_name}")


async def create_agent():
    """LangGraph Studio / API 入口工厂：尝试加载 MCP 工具，失败则仅用本地工具"""
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(_mcp_servers_config())
        mcp_tools = list(await client.get_tools())
    except Exception as e:
        print(f"[MCP] Studio 模式下 MCP 工具加载失败，仅使用本地工具: {e}")
        mcp_tools = []
    return build_graph(local_tools + mcp_tools).compile()


def get_runtime_agent():
    """返回当前可用的智能体：优先 MCP 增强版（启动时构建），回退本地工具版"""
    return _runtime_agent or _local_agent_with_memory


async def run_agent(user_message: str, session_id: str = "default") -> dict:
    """
    运行桌宠智能体（优先 MCP 混合版，回退本地版）
    :return: reply 最终回复 / audio_url 语音地址 / tool_calls 本次调用的工具列表
    """
    agent = get_runtime_agent()
    result = await agent.ainvoke(
        {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ]
        },
        config={"configurable": {"thread_id": session_id}},
    )

    messages = result["messages"]
    reply = messages[-1].content

    # 收集本次运行的工具调用情况
    audio_url = None
    used_tools = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                used_tools.append(tc["name"])
                if tc["name"] == "speak":
                    audio_url = "/static/tts/last.mp3"

    return {"reply": reply, "audio_url": audio_url, "tool_calls": used_tools}