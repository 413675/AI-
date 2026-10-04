import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import chat, asr
from app.core.config import settings

from contextlib import asynccontextmanager

from app.agents.pet_agent import close_runtime, init_runtime_agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时连接 MCP 服务器构建混合智能体，失败自动回退本地工具"""
    if settings.MCP_ENABLED:
        try:
            names = await init_runtime_agent()
            print(f"[MCP] 已连接，加载工具: {names}")
        except Exception as e:
            print(f"[MCP] 连接失败，回退本地工具: {e}")
    yield
    await close_runtime()


app = FastAPI(
    lifespan=lifespan,
    title="AI桌宠后端",
    description="AI Desktop Pet Backend - LangChain + LangGraph + FastAPI",
    version="0.0.1",
)

# CORS 配置 - 允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 桌面应用本地访问，开发阶段全开
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# temp 目录存放 TTS 等运行时产物（如 /static/tts/last.mp3）
TEMP_DIR = Path(__file__).resolve().parent.parent / "temp"
os.makedirs(TEMP_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(TEMP_DIR)), name="static")


@app.get("/")
async def root():
    return {"message": "AI桌宠后端运行中", "version": "0.0.1"}


@app.get("/health")
async def health():
    return {"status": "ok"}


# 注册路由
app.include_router(chat.router, prefix="/api/chat", tags=["聊天问答"])
app.include_router(asr.router, prefix="/api/asr", tags=["语音识别"])

