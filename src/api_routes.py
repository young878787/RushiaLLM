"""
API 路由定義
提供聊天、RAG 和系統管理的 RESTful API
"""

import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import aiofiles
import os
from pathlib import Path


# 請求/響應模型
class ChatRequest(BaseModel):
    message: str
    use_rag: bool = True
    stream: bool = False


class ChatResponse(BaseModel):
    response: str
    context_used: bool = False
    sources: List[str] = []


class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 5


class RAGSearchResponse(BaseModel):
    results: List[dict]
    total_found: int


class SystemStatus(BaseModel):
    status: str
    models_loaded: bool
    rag_ready: bool
    knowledge_base_stats: dict


def setup_routes(app, llm_manager, rag_system, config):
    """設置所有 API 路由"""
    
    logger = logging.getLogger(__name__)
    
    @app.get("/", response_model=dict)
    async def root():
        """根路徑 - 系統信息"""
        return {
            "name": "VTuber AI LLM API",
            "version": "1.0.0",
            "description": "本地化VTuber AI助手API",
            "endpoints": {
                "chat": "/chat",
                "rag_add": "/rag/add",
                "rag_search": "/rag/search",
                "health": "/health",
                "docs": "/docs"
            }
        }
    
    @app.get("/health", response_model=SystemStatus)
    async def health_check():
        """健康檢查"""
        try:
            models_loaded = (
                llm_manager.llm_model is not None and 
                llm_manager.embedding_model is not None
            )
            
            rag_ready = rag_system.collection is not None
            
            kb_stats = rag_system.get_stats()
            
            return SystemStatus(
                status="healthy" if models_loaded and rag_ready else "partial",
                models_loaded=models_loaded,
                rag_ready=rag_ready,
                knowledge_base_stats=kb_stats
            )
        except Exception as e:
            logger.error(f"健康檢查失敗: {e}")
            raise HTTPException(status_code=500, detail="系統檢查失敗")
    
    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """聊天對話端點"""
        try:
            context = ""
            sources = []
            context_used = False
            
            # 如果啟用 RAG，先搜索相關上下文
            if request.use_rag:
                context = await rag_system.get_context_for_query(request.message)
                if context:
                    context_used = True
                    # 提取來源信息
                    search_results = await rag_system.search(request.message)
                    sources = [
                        result['metadata'].get('filename', '未知來源') 
                        for result in search_results
                    ]
            
            # 生成回應
            response = await llm_manager.generate_response(
                prompt=request.message,
                context=context if context_used else None,
                stream=request.stream
            )
            
            return ChatResponse(
                response=response,
                context_used=context_used,
                sources=list(set(sources))  # 去重
            )
            
        except Exception as e:
            logger.error(f"聊天處理失敗: {e}")
            raise HTTPException(status_code=500, detail=f"聊天處理失敗: {str(e)}")
    
    @app.post("/rag/add")
    async def add_document(file: UploadFile = File(...)):
        """添加文檔到知識庫"""
        try:
            # 檢查文件類型
            allowed_extensions = {'.txt', '.pdf', '.docx'}
            file_extension = Path(file.filename).suffix.lower()
            
            if file_extension not in allowed_extensions:
                raise HTTPException(
                    status_code=400, 
                    detail=f"不支持的文件格式。支持的格式: {', '.join(allowed_extensions)}"
                )
            
            # 保存文件
            upload_dir = Path("data/documents")
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = upload_dir / file.filename
            
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # 添加到 RAG 系統
            success = await rag_system.add_document(str(file_path))
            
            if success:
                return {
                    "message": f"文檔 '{file.filename}' 已成功添加到知識庫",
                    "filename": file.filename,
                    "file_path": str(file_path)
                }
            else:
                # 如果處理失敗，刪除文件
                if file_path.exists():
                    file_path.unlink()
                raise HTTPException(status_code=500, detail="文檔處理失敗")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"添加文檔失敗: {e}")
            raise HTTPException(status_code=500, detail=f"添加文檔失敗: {str(e)}")
    
    @app.post("/rag/search", response_model=RAGSearchResponse)
    async def search_knowledge_base(request: RAGSearchRequest):
        """搜索知識庫"""
        try:
            results = await rag_system.search(request.query, request.top_k)
            
            return RAGSearchResponse(
                results=results,
                total_found=len(results)
            )
            
        except Exception as e:
            logger.error(f"知識庫搜索失敗: {e}")
            raise HTTPException(status_code=500, detail=f"搜索失敗: {str(e)}")
    
    @app.get("/rag/stats")
    async def get_knowledge_base_stats():
        """獲取知識庫統計信息"""
        try:
            stats = rag_system.get_stats()
            
            # 添加更多統計信息
            docs_dir = Path("data/documents")
            file_count = len(list(docs_dir.glob("**/*"))) if docs_dir.exists() else 0
            
            return {
                **stats,
                "uploaded_files": file_count,
                "supported_formats": [".txt", ".pdf", ".docx"]
            }
            
        except Exception as e:
            logger.error(f"獲取統計信息失敗: {e}")
            raise HTTPException(status_code=500, detail="獲取統計信息失敗")
    
    @app.delete("/rag/clear")
    async def clear_knowledge_base():
        """清空知識庫"""
        try:
            success = await rag_system.clear_knowledge_base()
            
            if success:
                return {"message": "知識庫已清空"}
            else:
                raise HTTPException(status_code=500, detail="清空知識庫失敗")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"清空知識庫失敗: {e}")
            raise HTTPException(status_code=500, detail=f"清空知識庫失敗: {str(e)}")
    
    @app.get("/config")
    async def get_config():
        """獲取系統配置（敏感信息已隱藏）"""
        safe_config = {
            "vtuber": config["vtuber"],
            "api": {
                "host": config["api"]["host"],
                "port": config["api"]["port"]
            },
            "models": {
                "llm": {
                    "quantization": config["models"]["llm"]["quantization"],
                    "max_length": config["models"]["llm"]["max_length"]
                }
            }
        }
        return safe_config