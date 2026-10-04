"""RAG 服务：Milvus 向量库 + 阿里云 Embedding，为桌宠提供知识库问答。

- 懒加载：首次检索/入库时才连接 Milvus，服务未启动不影响后端启动
- 自动建 collection（pet_knowledge：id/vector/text/source，COSINE + AUTOINDEX）
- 文档入库：RecursiveCharacterTextSplitter 切块（500 字 / 重叠 50）
"""
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings

COLLECTION = "pet_knowledge"


class RAGService:
    def __init__(self):
        self._client = None
        self._embeddings = None
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # ---------- 基础组件（懒加载） ----------

    def _get_embeddings(self) -> OpenAIEmbeddings:
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                api_key=settings.LLM_API_KEY,
                base_url=settings.EMBEDDING_BASE_URL or settings.LLM_BASE_URL,
                dimensions=settings.VECTOR_DIM,
                # 非 OpenAI 端点必须关掉 tiktoken 分块，否则会报编码错误
                check_embedding_ctx_length=False,
            )
        return self._embeddings

    def _get_client(self):
        if self._client is None:
            from pymilvus import MilvusClient

            self._client = MilvusClient(uri=settings.VECTOR_URL)
        return self._client

    def _ensure_collection(self):
        """连接 Milvus 并确保 collection 存在（首次自动创建）"""
        from pymilvus import DataType

        client = self._get_client()
        if not client.has_collection(COLLECTION):
            schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=settings.VECTOR_DIM)
            schema.add_field("text", DataType.VARCHAR, max_length=8192)
            schema.add_field("source", DataType.VARCHAR, max_length=512)
            index_params = client.prepare_index_params()
            index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
            client.create_collection(COLLECTION, schema=schema, index_params=index_params)
            print(f"[RAG] 已创建向量集合 {COLLECTION}（dim={settings.VECTOR_DIM}）")
        return client

    # ---------- 对外能力 ----------

    def add_texts(self, texts: list[str], source: str = "manual") -> int:
        """文本切块后向量化入库，返回入库条数"""
        if not texts:
            return 0
        embs = self._get_embeddings().embed_documents(texts)
        client = self._ensure_collection()
        rows = [{"vector": e, "text": t, "source": source} for t, e in zip(texts, embs)]
        client.insert(COLLECTION, rows)
        return len(rows)

    def ingest_file(self, path: str) -> dict:
        """读取 .md/.txt 文件，切块入库"""
        p = Path(path)
        if not p.exists():
            return {"error": f"文件不存在: {path}"}
        text = p.read_text(encoding="utf-8-sig")
        if not text.strip():
            return {"error": "文件内容为空"}
        chunks = self._splitter.split_text(text)
        count = self.add_texts(chunks, source=p.name)
        return {"file": p.name, "chunks": count}

    def search(self, query: str, k: int = 4) -> list[dict]:
        """向量检索最相关的 k 条知识片段"""
        emb = self._get_embeddings().embed_query(query)
        client = self._ensure_collection()
        res = client.search(
            COLLECTION, data=[emb], limit=k, output_fields=["text", "source"]
        )
        return [
            {"text": h["entity"]["text"], "source": h["entity"]["source"], "score": h["distance"]}
            for h in res[0]
        ]


rag_service = RAGService()