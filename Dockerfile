FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

# 用本地 .venv 导出的完整锁定清单安装，--no-deps 跳过依赖解析，
# 保证容器内版本与本地实测可运行环境完全一致（本地 langchain 0.3.30 + core 1.6.6
# 是 pip 强装出来的组合，正常解析会冲突，因此必须锁死版本）
COPY requirements.lock.txt .
RUN pip install --no-cache-dir --no-deps -r requirements.lock.txt \
    -i https://mirrors.aliyun.com/pypi/simple/

# 再拷代码（.dockerignore 已排除 .venv/.git/__pycache__ 等）
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
