import httpx
from app.core.config import settings


class ASRService:
    """
    语音识别服务
    当前走阿里云 DashScope 兼容接口（与 LLM 同一个 key），
    预留百度 ASR 接口，通过 settings.ASR_PROVIDER 切换
    """

    async def transcribe(self, audio_bytes: bytes, filename: str = "speech.wav") -> str:
        """
        语音转文字
        :param audio_bytes: WAV 音频二进制数据
        :return: 识别出的文本
        """
        provider = settings.ASR_PROVIDER
        if not provider:
            raise RuntimeError("ASR 未启用，请在 .env 中配置 ASR_PROVIDER")
        if provider == "dashscope":
            return await self._transcribe_dashscope(audio_bytes, filename)
        elif provider == "baidu":
            return await self._transcribe_baidu(audio_bytes, filename)
        raise ValueError(f"不支持的 ASR provider: {provider}")

    async def _transcribe_dashscope(self, audio_bytes: bytes, filename: str) -> str:
        """阿里云 DashScope 兼容接口（POST /audio/asr）"""
        url = settings.ASR_BASE_URL or settings.LLM_BASE_URL.rstrip("/") + "/audio/asr"
        headers = {"Authorization": f"Bearer {settings.LLM_API_KEY}"}
        files = {"file": (filename, audio_bytes, "audio/wav")}
        data = {"model": settings.ASR_MODEL}

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, headers=headers, files=files, data=data)
            result = resp.json()

        if "text" in result:
            return result["text"]
        raise RuntimeError(f"ASR 返回异常: {result}")

    async def _transcribe_baidu(self, audio_bytes: bytes, filename: str) -> str:
        """百度 ASR（预留接口，待对接）"""
        raise NotImplementedError("百度 ASR 暂未对接，请使用 dashscope provider")


asr_service = ASRService()
