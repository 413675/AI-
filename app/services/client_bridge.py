"""客户端工具桥：让服务器上的 Agent 操作用户电脑上的文件。

原理：桌宠客户端（Electron 主进程）启动时通过 WebSocket 连到后端
/ws/client-tools 注册自己；Agent 需要读写「用户电脑」文件时，工具函数
把请求经该连接推给客户端，客户端在本地执行（写操作弹窗确认、目录白名单）
后把结果回传，工具拿到结果继续推理。

- 每个 client_id 一条连接（当前单用户场景用 default）
- 请求-响应用 request_id 配对，支持并发调用
- 客户端断线时，其所有挂起请求立即失败，工具拿到明确错误而不是傻等超时
"""
import asyncio
import json
import uuid
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect


class ClientBridge:
    def __init__(self):
        self._clients: dict[str, WebSocket] = {}
        self._pending: dict[str, tuple[asyncio.Future, str]] = {}  # request_id -> (future, client_id)
        self._lock = asyncio.Lock()

    # ---------- 连接管理 ----------

    @property
    def connected_clients(self) -> list[str]:
        return list(self._clients.keys())

    def is_connected(self, client_id: str = "default") -> bool:
        return client_id in self._clients

    async def serve(self, ws: WebSocket, client_id: str) -> None:
        """注册连接并持续接收客户端回包，直到断开"""
        async with self._lock:
            old = self._clients.get(client_id)
            self._clients[client_id] = ws
        if old is not None:
            try:
                await old.close()
            except Exception:
                pass

        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                rid = msg.get("request_id")
                entry = self._pending.get(rid) if rid else None
                if entry:
                    fut, _ = entry
                    if not fut.done():
                        fut.set_result(msg.get("result"))
        except WebSocketDisconnect:
            pass
        finally:
            async with self._lock:
                if self._clients.get(client_id) is ws:
                    self._clients.pop(client_id, None)
            # 客户端掉线：让它名下所有挂起请求立即失败
            for rid, (fut, owner) in list(self._pending.items()):
                if owner == client_id and not fut.done():
                    fut.set_exception(RuntimeError("客户端连接已断开"))
                    self._pending.pop(rid, None)

    # ---------- 工具调用 ----------

    async def call(
        self, tool: str, args: dict[str, Any], client_id: str = "default", timeout: float = 90.0
    ) -> dict[str, Any]:
        """向指定客户端转发工具调用并等待结果。

        :return: 客户端回包 {"ok": bool, "data"|"error": ...}
        :raises RuntimeError: 客户端未连接 / 断线 / 超时
        """
        ws = self._clients.get(client_id)
        if ws is None:
            raise RuntimeError(
                f"桌宠客户端（client_id={client_id}）未连接，无法操作你的电脑。"
                "请确认桌宠客户端已启动且已连上后端。"
            )

        rid = uuid.uuid4().hex
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[rid] = (fut, client_id)
        try:
            await ws.send_text(json.dumps({"request_id": rid, "tool": tool, "args": args}))
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(f"客户端执行 {tool} 超时（{timeout:.0f}s）") from None
        finally:
            self._pending.pop(rid, None)


client_bridge = ClientBridge()
