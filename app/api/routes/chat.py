import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.agents.pet_agent import run_agent

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    audio_url: str | None = None
    tool_calls: list[str] = []


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """智能体入口：LangGraph 驱动，自动决定是否调用工具（天气/爬虫/文件/语音）"""
    return await run_agent(req.message, req.session_id)


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    """流式输出 AI 回复（SSE，token 级）。

    使用 astream_events(version="v2") 而非 astream()：
    astream 按图节点吐事件（LLM 整段生成完才来一坨，等于没有流式），
    astream_events 能拿到 on_chat_model_stream 的 token 增量，实现逐字输出。
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from app.agents.pet_agent import SYSTEM_PROMPT, get_runtime_agent

    async def event_generator():
        full_content = ""
        tool_calls: list[str] = []
        audio_url = None

        def sse(payload: dict) -> str:
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        try:
            async for event in get_runtime_agent().astream_events(
                {
                    "messages": [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(content=req.message),
                    ]
                },
                config={"configurable": {"thread_id": req.session_id}},
                version="v2",
            ):
                et = event["event"]

                # token 级增量：LLM 每生成一个片段就推给前端
                if et == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if chunk is None:
                        continue
                    content = getattr(chunk, "content", None)
                    if isinstance(content, str) and content:
                        full_content += content
                        yield sse({"type": "delta", "content": content})
                    # 工具调用片段：LLM 决定调用工具时立即通知前端
                    for tcc in getattr(chunk, "tool_call_chunks", None) or []:
                        name = (
                            tcc.get("name")
                            if isinstance(tcc, dict)
                            else getattr(tcc, "name", None)
                        )
                        if name and name not in tool_calls:
                            tool_calls.append(name)
                            yield sse({"type": "tool", "name": name})
                        if name == "speak":
                            audio_url = "/static/tts/last.mp3"

                # 工具开始执行（工具碎片未捕获时兜底，如 MCP 工具）
                elif et == "on_tool_start":
                    name = event.get("name", "")
                    if name and name not in tool_calls:
                        tool_calls.append(name)
                    if name == "speak":
                        audio_url = "/static/tts/last.mp3"
                    yield sse({"type": "tool", "name": name})

        except Exception as e:
            yield sse({"type": "error", "message": str(e)})
            return

        yield sse(
            {
                "type": "done",
                "content": full_content,
                "tool_calls": tool_calls,
                "audio_url": audio_url,
            }
        )

    return StreamingResponse(event_generator(), media_type="text/event-stream")