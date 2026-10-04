import asyncio
import edge_tts
from typing import Optional
from app.core.config import settings


class TTSService:
    """
    语音合成服务
    当前支持 Edge-TTS，预留百度 TTS 接口
    通过 settings.TTS_PROVIDER 切换
    """

    def __init__(self):
        self.provider = settings.TTS_PROVIDER

    async def synthesize(self, text: str, output_path: str, voice: Optional[str] = None) -> str:
        """
        合成语音并保存到文件
        :param text: 要合成的文本
        :param output_path: 输出音频文件路径
        :param voice: 语音角色，默认从配置读取
        :return: 输出文件路径
        """
        if self.provider == "edge":
            return await self._synthesize_edge(text, output_path, voice)
        elif self.provider == "baidu":
            return await self._synthesize_baidu(text, output_path, voice)
        else:
            raise ValueError(f"不支持的 TTS provider: {self.provider}")

    async def _synthesize_edge(self, text: str, output_path: str, voice: Optional[str] = None) -> str:
        """Edge-TTS 合成（免费，无需 API Key）"""
        voice_name = voice or settings.EDGE_TTS_VOICE
        communicate = edge_tts.Communicate(text, voice_name)
        await communicate.save(output_path)
        return output_path

    async def _synthesize_baidu(self, text: str, output_path: str, voice: Optional[str] = None) -> str:
        """
        百度 TTS 合成（预留接口，待对接）
        需要 BAIDU_TTS_APP_ID / API_KEY / SECRET_KEY
        """
        # TODO: 对接百度 TTS API
        # from aip import AipSpeech
        # client = AipSpeech(settings.BAIDU_TTS_APP_ID, settings.BAIDU_TTS_API_KEY, settings.BAIDU_TTS_SECRET_KEY)
        # result = client.synthesis(text, 'zh', 1, {'vol': 5, 'per': 0})
        # with open(output_path, 'wb') as f:
        #     f.write(result)
        raise NotImplementedError("百度 TTS 暂未对接，请使用 edge provider")


tts_service = TTSService()
