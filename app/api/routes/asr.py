from fastapi import APIRouter, File, HTTPException, UploadFile
from app.services.asr_service import asr_service

router = APIRouter()


@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """上传录音文件，返回识别文本"""
    try:
        audio = await file.read()
        text = await asr_service.transcribe(audio, file.filename or "speech.wav")
        return {"text": text}
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音识别失败: {e}")
