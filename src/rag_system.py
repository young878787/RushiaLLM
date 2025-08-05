"""
RAG (檢索增強生成) 系統
負責文檔處理、向量存儲和檢索
"""

import logging
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
import numpy as np


class RAGSystem:
    def __init__(self, config: dict, embedding_model):
        self.config = config
        self.embedding_model = embedding_model
        self.logger = logging.getLogger(__name__)
        
        # RAG 配置
        self.rag_config = config['rag']
        
        # 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.rag_config['retrieval']['chunk_size'],
            chunk_overlap=self.rag_config['retrieval']['chunk_overlap'],
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "]
        )
        
        # 向量數據庫
        self.chroma_client = None
        self.collection = None
        
        # 語義向量增強系統
        self.semantic_vectors = {}
        self.vector_cache_initialized = False
        
        # 多樣性控制配置
        self.diversity_config = {
            'max_per_file': 2,  # 每個文件最多2個結果
            'min_files': 3,     # 至少來自3個不同文件
            'semantic_analysis_limit': 1  # semantic_analysis.md最多1個結果
        }
        
        # Rushia Knowledge 分類系統 (平衡權重)
        self.knowledge_categories = {
            'core_identity': {'priority': 'high', 'weight': 1.4},
            'vocabulary': {'priority': 'high', 'weight': 1.3},
            'content': {'priority': 'high', 'weight': 1.4},  # 降低content優先級
            'relationships': {'priority': 'medium', 'weight': 1.2},
            'timeline': {'priority': 'low', 'weight': 1.0}
        }
    
    async def initialize(self):
        """初始化 RAG 系統"""
        await self._setup_vector_db()
        await self._load_existing_documents()
        await self._load_rushia_knowledge()
        await self._initialize_semantic_vectors()
    
    async def _setup_vector_db(self):
        """設置向量數據庫"""
        self.logger.info("初始化向量數據庫...")

        # 取得 LLM 目錄（rag_system.py 的上兩層）
        llm_dir = Path(__file__).parent.parent.resolve()
        persist_dir = llm_dir / Path(self.rag_config['vector_db']['persist_directory'])
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化 ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 獲取或創建集合
        collection_name = self.rag_config['vector_db']['collection_name']
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
            self.logger.info(f"載入現有集合: {collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "VTuber AI 知識庫 + Rushia Knowledge"}
            )
            self.logger.info(f"創建新集合: {collection_name}")
    
    async def _load_existing_documents(self):
        """載入現有文檔"""
        docs_dir = Path("data/documents")
        if not docs_dir.exists():
            docs_dir.mkdir(parents=True, exist_ok=True)
            return
        
        # 檢查是否有新文檔需要處理
        doc_files = list(docs_dir.glob("**/*"))
        doc_files = [f for f in doc_files if f.is_file() and f.suffix.lower() in ['.txt', '.pdf', '.docx']]
        
        if doc_files:
            self.logger.info(f"發現 {len(doc_files)} 個文檔，開始處理...")
            for doc_file in doc_files:
                await self.add_document(str(doc_file))
    
    async def _load_rushia_knowledge(self):
        """載入Rushia Knowledge markdown文件"""
        knowledge_dir = Path("rushia_wiki/rushia_knowledge")
        if not knowledge_dir.exists():
            self.logger.warning(f"Rushia Knowledge目錄不存在: {knowledge_dir}")
            return
        
        # 獲取所有markdown文件
        md_files = list(knowledge_dir.glob("**/*.md"))
        
        if md_files:
            self.logger.info(f"🧠 發現 {len(md_files)} 個Rushia Knowledge文件，開始整合...")
            
            for md_file in md_files:
                await self._add_knowledge_file(md_file)
            
            self.logger.info("✅ Rushia Knowledge整合完成")
        else:
            self.logger.warning("未找到Rushia Knowledge markdown文件")
    
    async def _add_knowledge_file(self, file_path: Path):
        """添加知識文件到RAG系統"""
        try:
            self.logger.info(f"處理知識文件: {file_path.name}")
            
            # 讀取markdown內容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                self.logger.warning(f"文件內容為空: {file_path}")
                return False
            
            # 分析文件類型和優先級
            category = self._categorize_knowledge_file(file_path)
            priority_info = self.knowledge_categories.get(category, {'priority': 'medium', 'weight': 1.0})
            
            # 智能分塊
            chunks = self._smart_chunk_markdown(content, file_path)
            
            if not chunks:
                return False
            
            # 生成嵌入向量
            texts = [chunk['content'] for chunk in chunks]
            embeddings = self.embedding_model.encode(texts).tolist()
            
            # 準備數據
            ids = [f"rushia_{file_path.stem}_{i}" for i in range(len(chunks))]
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                metadata = {
                    'source': str(file_path),
                    'filename': file_path.name,
                    'file_type': 'markdown',
                    'category': category,
                    'priority': priority_info['priority'],
                    'weight': priority_info['weight'],
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'knowledge_type': 'rushia_knowledge',
                    **chunk.get('metadata', {})
                }
                metadatas.append(metadata)
            
            # 添加到向量數據庫
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"✅ 成功添加 {len(chunks)} 個知識塊: {file_path.name} ({category})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 添加知識文件失敗 {file_path}: {e}")
            return False
    
    def _categorize_knowledge_file(self, file_path: Path) -> str:
        """根據文件路徑分類知識文件"""
        path_parts = file_path.parts
        
        for part in path_parts:
            if part in self.knowledge_categories:
                return part
        
        return 'general'
    
    def _smart_chunk_markdown(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """智能分塊markdown內容"""
        chunks = []
        
        # 按標題分割
        sections = self._split_by_headers(content)
        
        for section in sections:
            if len(section['content']) > self.rag_config['retrieval']['chunk_size']:
                # 大段落需要進一步分割
                sub_chunks = self.text_splitter.split_text(section['content'])
                for i, sub_chunk in enumerate(sub_chunks):
                    chunks.append({
                        'content': sub_chunk,
                        'metadata': {
                            'section_title': section['title'],
                            'section_level': section['level'],
                            'sub_chunk': i
                        }
                    })
            else:
                # 小段落保持完整
                chunks.append({
                    'content': section['content'],
                    'metadata': {
                        'section_title': section['title'],
                        'section_level': section['level']
                    }
                })
        
        return chunks
    
    def _split_by_headers(self, content: str) -> List[Dict[str, Any]]:
        """按markdown標題分割內容"""
        lines = content.split('\n')
        sections = []
        current_section = {'title': '', 'level': 0, 'content': ''}
        
        for line in lines:
            if line.startswith('#'):
                # 保存前一個section
                if current_section['content'].strip():
                    sections.append(current_section.copy())
                
                # 開始新section
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                current_section = {
                    'title': title,
                    'level': level,
                    'content': line + '\n'
                }
            else:
                current_section['content'] += line + '\n'
        
        # 添加最後一個section
        if current_section['content'].strip():
            sections.append(current_section)
        
        return sections
    
    async def add_document(self, file_path: str) -> bool:
        """添加文檔到知識庫"""
        try:
            self.logger.info(f"處理文檔: {file_path}")
            
            # 載入文檔
            documents = await self._load_document(file_path)
            if not documents:
                return False
            
            # 分割文本
            chunks = []
            for doc in documents:
                doc_chunks = self.text_splitter.split_text(doc['content'])
                for i, chunk in enumerate(doc_chunks):
                    chunks.append({
                        'content': chunk,
                        'metadata': {
                            **doc['metadata'],
                            'chunk_id': i,
                            'total_chunks': len(doc_chunks)
                        }
                    })
            
            if not chunks:
                return False
            
            # 生成嵌入向量
            texts = [chunk['content'] for chunk in chunks]
            embeddings = self.embedding_model.encode(texts).tolist()
            
            # 準備數據
            ids = [f"{Path(file_path).stem}_{i}" for i in range(len(chunks))]
            metadatas = [chunk['metadata'] for chunk in chunks]
            
            # 添加到向量數據庫
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"✅ 成功添加 {len(chunks)} 個文本塊到知識庫")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 添加文檔失敗: {e}")
            return False
    
    async def _load_document(self, file_path: str) -> List[Dict[str, Any]]:
        """載入文檔內容"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            self.logger.error(f"文件不存在: {file_path}")
            return []
        
        try:
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() == '.docx':
                loader = Docx2txtLoader(str(file_path))
            elif file_path.suffix.lower() == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
            else:
                self.logger.warning(f"不支持的文件格式: {file_path.suffix}")
                return []
            
            docs = loader.load()
            
            result = []
            for doc in docs:
                result.append({
                    'content': doc.page_content,
                    'metadata': {
                        'source': str(file_path),
                        'filename': file_path.name,
                        'file_type': file_path.suffix.lower(),
                        'category': 'traditional_document',
                        'priority': 'medium',
                        'weight': 1.0,
                        'knowledge_type': 'traditional_document',
                        **doc.metadata
                    }
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"載入文檔失敗 {file_path}: {e}")
            return []
    
    async def semantic_enhanced_search(self, query: str, emotional_analysis: Dict[str, Any] = None, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """語義向量增強搜索 - 基於向量調整的智能檢索"""
        if top_k is None:
            top_k = self.rag_config['retrieval']['top_k']
        
        try:
            # 確保語義向量已初始化
            if not self.vector_cache_initialized:
                await self._initialize_semantic_vectors()
            
            # 獲取原始查詢向量
            query_vector = self.embedding_model.encode([query])[0]
            
            # 根據情感分析調整查詢向量
            adjusted_vector = self._adjust_query_vector(query_vector, emotional_analysis)
            
            # 執行多向量搜索
            results = await self._multi_vector_search(adjusted_vector, emotional_analysis, top_k)
            
            # 確保結果多樣性
            diverse_results = self._ensure_result_diversity(results)
            
            # 記錄語義搜索日誌
            self._log_semantic_search_results(query, emotional_analysis, diverse_results)
            
            return diverse_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"語義增強搜索失敗: {e}")
            # 回退到標準搜索
            return await self.search(query, top_k)

    async def search(self, query: str, top_k: Optional[int] = None, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """增強版搜索 - 支援分類過濾和優先級權重"""
        if top_k is None:
            top_k = self.rag_config['retrieval']['top_k']
        
        try:
            # 生成查詢嵌入
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # 構建過濾條件
            where_filter = {}
            if category_filter:
                where_filter['category'] = category_filter
            
            # 搜索 (獲取更多結果用於重新排序)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2,
                include=['documents', 'metadatas', 'distances'],
                where=where_filter if where_filter else None
            )
            
            # 格式化和重新排序結果
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    similarity = 1 - results['distances'][0][i]
                    metadata = results['metadatas'][0][i]
                    
                    # 過濾低相似度結果
                    if similarity >= self.rag_config['retrieval']['similarity_threshold']:
                        # 計算加權分數
                        weight = metadata.get('weight', 1.0)
                        weighted_score = similarity * weight
                        
                        search_results.append({
                            'content': results['documents'][0][i],
                            'metadata': metadata,
                            'similarity': similarity,
                            'weighted_score': weighted_score
                        })
            
            # 按加權分數排序
            search_results.sort(key=lambda x: x['weighted_score'], reverse=True)
            
            # 返回top_k結果
            final_results = search_results[:top_k]
            
            # 詳細的搜索結果日誌
            if final_results:
                sources = list(set([r['metadata'].get('filename', 'unknown') for r in final_results]))
                categories = list(set([r['metadata'].get('category', 'unknown') for r in final_results]))
                similarities = [r['similarity'] for r in final_results]
                
                self.logger.info(f"🔍 增強搜索完成: {len(final_results)} 個結果")
                self.logger.info(f"   📚 來源文件: {', '.join(sources)}")
                self.logger.info(f"   📂 涉及類別: {', '.join(categories)}")
                self.logger.info(f"   📊 相似度範圍: {min(similarities):.3f} - {max(similarities):.3f}")
                if category_filter:
                    self.logger.info(f"   🎯 類別過濾: {category_filter}")
            else:
                self.logger.info(f"🔍 增強搜索完成: 0 個結果 (查詢: '{query}')")
                self.logger.info(f"   ⚠️ 可能原因: 相似度低於閾值 {self.rag_config['retrieval']['similarity_threshold']}")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"搜索失敗: {e}")
            return []
    
    async def get_context_for_query(self, query: str) -> str:
        """為查詢獲取增強版上下文 (不記錄日誌，避免重複)"""
        try:
            # 生成查詢嵌入
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # 搜索 (不使用 search 方法，避免重複日誌)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=5 * 2,
                include=['documents', 'metadatas', 'distances']
            )
            
            # 格式化結果 (簡化版，不記錄日誌)
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    similarity = 1 - results['distances'][0][i]
                    metadata = results['metadatas'][0][i]
                    
                    if similarity >= self.rag_config['retrieval']['similarity_threshold']:
                        weight = metadata.get('weight', 1.0)
                        weighted_score = similarity * weight
                        
                        search_results.append({
                            'content': results['documents'][0][i],
                            'metadata': metadata,
                            'similarity': similarity,
                            'weighted_score': weighted_score
                        })
            
            # 排序並取前5個
            search_results.sort(key=lambda x: x['weighted_score'], reverse=True)
            final_results = search_results[:5]
            
            if not final_results:
                return ""
            
            # 組合上下文
            context_parts = []
            for result in final_results:
                content = result['content'].strip()
                metadata = result['metadata']
                
                # 構建來源信息
                source_info = []
                if metadata.get('category'):
                    source_info.append(f"類別: {metadata['category']}")
                if metadata.get('section_title'):
                    source_info.append(f"章節: {metadata['section_title']}")
                if metadata.get('filename'):
                    source_info.append(f"文件: {metadata['filename']}")
                
                source_str = " | ".join(source_info) if source_info else "未知來源"
                similarity = result['similarity']
                
                context_parts.append(f"[{source_str} | 相關度: {similarity:.2f}]\n{content}")
            
            return "\n\n---\n\n".join(context_parts)
            
        except Exception as e:
            self.logger.error(f"獲取上下文失敗: {e}")
            return ""
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取知識庫統計信息"""
        try:
            count = self.collection.count()
            
            # 統計不同類型的文檔數量
            try:
                all_results = self.collection.get(include=['metadatas'])
                category_counts = {}
                knowledge_type_counts = {}
                
                if all_results['metadatas']:
                    for metadata in all_results['metadatas']:
                        category = metadata.get('category', 'unknown')
                        knowledge_type = metadata.get('knowledge_type', 'unknown')
                        
                        category_counts[category] = category_counts.get(category, 0) + 1
                        knowledge_type_counts[knowledge_type] = knowledge_type_counts.get(knowledge_type, 0) + 1
                
                return {
                    "total_documents": count,
                    "collection_name": self.rag_config['vector_db']['collection_name'],
                    "category_breakdown": category_counts,
                    "knowledge_type_breakdown": knowledge_type_counts,
                    "knowledge_categories": list(self.knowledge_categories.keys()),
                    "enhanced_features": ["markdown_support", "category_filtering", "priority_weighting"]
                }
            except:
                return {
                    "total_documents": count,
                    "collection_name": self.rag_config['vector_db']['collection_name']
                }
        except Exception as e:
            self.logger.error(f"獲取統計信息失敗: {e}")
            return {"total_documents": 0, "collection_name": "unknown"}
    
    async def clear_knowledge_base(self):
        """清空知識庫"""
        try:
            # 刪除現有集合
            collection_name = self.rag_config['vector_db']['collection_name']
            self.chroma_client.delete_collection(collection_name)
            
            # 重新創建集合
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "VTuber AI 知識庫"}
            )
            
            self.logger.info("知識庫已清空")
            return True
            
        except Exception as e:
            self.logger.error(f"清空知識庫失敗: {e}")
            return False
    
    def _build_semantic_query(self, query: str, emotional_analysis: Dict[str, Any] = None) -> str:
        """構建語義增強查詢"""
        if not emotional_analysis:
            return query
        
        query_parts = [query]
        
        # 添加情感相關關鍵詞
        emotional_intensity = emotional_analysis.get('emotional_intensity', 'mild')
        if emotional_intensity == 'very_strong':
            query_parts.append('強烈負面情緒 溫柔安慰 心疼關心')
        elif emotional_intensity == 'moderate':
            query_parts.append('中等負面情緒 溫柔回應 關心理解')
        elif emotional_intensity == 'mild':
            query_parts.append('平靜情緒 自然親切 可愛風格')
        
        # 添加親密度相關關鍵詞
        intimacy_score = emotional_analysis.get('intimacy_score', 0.0)
        if intimacy_score > 2.5:
            query_parts.append('深度親密 戀人甜蜜 親密稱呼')
        elif intimacy_score > 1.5:
            query_parts.append('親密階段 撒嬌黏人 溫暖互動')
        elif intimacy_score > 0.5:
            query_parts.append('熟悉階段 親切友善 溫暖關係')
        else:
            query_parts.append('初始階段 友善距離 建立關係')
        
        # 添加意圖相關關鍵詞
        detected_intent = emotional_analysis.get('detected_intent', '')
        intent_keywords = {
            'companionship_request': '尋求陪伴 開心願意 一起互動',
            'emotional_support': '尋求安慰 溫柔體貼 關懷照顧',
            'intimate_expression': '表達愛意 害羞開心 甜蜜回應',
            'question_asking': '詢問問題 耐心解答 樂於助人',
            'casual_chat': '日常閒聊 活潑自然 輕鬆互動'
        }
        
        if detected_intent in intent_keywords:
            query_parts.append(intent_keywords[detected_intent])
        
        return ' '.join(query_parts)
    
    def _determine_semantic_search_strategy(self, emotional_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """確定語義搜索策略"""
        if not emotional_analysis:
            return {
                'primary_category': 'content',
                'boost_keywords': [],
                'priority_weight': 1.0
            }
        
        # 基於情感強度確定策略
        emotional_intensity = emotional_analysis.get('emotional_intensity', 'mild')
        detected_intent = emotional_analysis.get('detected_intent', '')
        intimacy_score = emotional_analysis.get('intimacy_score', 0.0)
        
        strategy = {
            'primary_category': 'content',  # semantic_analysis.md 在 content 目錄
            'boost_keywords': [],
            'priority_weight': 1.6  # content 類別的權重
        }
        
        # 根據情感強度調整
        if emotional_intensity == 'very_strong':
            strategy['boost_keywords'].extend(['強烈負面情緒', '溫柔安慰', '沒關係', '我在這裡'])
            strategy['priority_weight'] = 1.8
        elif emotional_intensity == 'moderate':
            strategy['boost_keywords'].extend(['中等負面情緒', '溫柔回應', '關心'])
            strategy['priority_weight'] = 1.7
        
        # 根據意圖調整
        if detected_intent == 'emotional_support':
            strategy['boost_keywords'].extend(['安慰', '體貼', '關懷'])
        elif detected_intent == 'intimate_expression':
            strategy['boost_keywords'].extend(['愛意', '害羞', '甜蜜'])
        elif detected_intent == 'companionship_request':
            strategy['boost_keywords'].extend(['陪伴', '一起', '開心'])
        
        # 根據親密度調整
        if intimacy_score > 2.0:
            strategy['boost_keywords'].extend(['親愛的', '寶貝', '撒嬌'])
        
        return strategy
    
    async def _multi_tier_search(self, enhanced_query: str, search_strategy: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        """多層次搜索"""
        all_results = []
        
        # 第一層：優先搜索 content 類別（包含 semantic_analysis.md）
        content_results = await self.search(
            enhanced_query, 
            top_k=top_k * 2, 
            category_filter=search_strategy['primary_category']
        )
        
        # 為 content 結果添加優先級加成
        for result in content_results:
            result['tier'] = 'primary'
            result['tier_boost'] = 0.2
            result['boosted_score'] = result.get('weighted_score', result['similarity']) + result['tier_boost']
        
        all_results.extend(content_results)
        
        # 第二層：搜索其他高優先級類別
        if len(content_results) < top_k:
            other_categories = ['vocabulary', 'core_identity']
            for category in other_categories:
                other_results = await self.search(
                    enhanced_query,
                    top_k=max(2, top_k - len(content_results)),
                    category_filter=category
                )
                
                # 為其他結果添加較低的加成
                for result in other_results:
                    result['tier'] = 'secondary'
                    result['tier_boost'] = 0.1
                    result['boosted_score'] = result.get('weighted_score', result['similarity']) + result['tier_boost']
                
                all_results.extend(other_results)
        
        return all_results
    
    def _semantic_rerank(self, results: List[Dict[str, Any]], emotional_analysis: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """基於語義分析重新排序結果"""
        if not emotional_analysis or not results:
            return results
        
        # 獲取語義關鍵詞
        boost_keywords = []
        
        emotional_intensity = emotional_analysis.get('emotional_intensity', 'mild')
        detected_intent = emotional_analysis.get('detected_intent', '')
        intimacy_score = emotional_analysis.get('intimacy_score', 0.0)
        
        # 收集所有相關關鍵詞
        if emotional_intensity == 'very_strong':
            boost_keywords.extend(['強烈負面情緒', '溫柔安慰', '心疼', '沒關係', '我在這裡'])
        elif emotional_intensity == 'moderate':
            boost_keywords.extend(['中等負面情緒', '溫柔回應', '關心', '理解'])
        
        if detected_intent == 'emotional_support':
            boost_keywords.extend(['安慰', '體貼', '關懷', '陪伴'])
        elif detected_intent == 'intimate_expression':
            boost_keywords.extend(['愛意', '害羞', '甜蜜', '討厭啦'])
        
        if intimacy_score > 2.0:
            boost_keywords.extend(['親愛的', '寶貝', '撒嬌', '戀人'])
        elif intimacy_score > 1.0:
            boost_keywords.extend(['親近', '溫暖', '可愛'])
        
        # 為每個結果計算語義匹配分數
        for result in results:
            content = result['content'].lower()
            semantic_boost = 0.0
            
            # 計算關鍵詞匹配度
            matched_keywords = 0
            for keyword in boost_keywords:
                if keyword in content:
                    matched_keywords += 1
            
            if boost_keywords:
                semantic_boost = (matched_keywords / len(boost_keywords)) * 0.3
            
            # 檢查是否來自 semantic_analysis.md
            if 'semantic_analysis' in result['metadata'].get('filename', ''):
                semantic_boost += 0.2
            
            # 更新最終分數
            base_score = result.get('boosted_score', result.get('weighted_score', result['similarity']))
            result['final_score'] = base_score + semantic_boost
            result['semantic_boost'] = semantic_boost
        
        # 按最終分數排序
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return results
    
    def _log_semantic_search_results(self, query: str, emotional_analysis: Dict[str, Any], results: List[Dict[str, Any]]):
        """記錄語義搜索結果"""
        if not results:
            self.logger.info(f"🔍 語義增強搜索: 0 個結果 (查詢: '{query}')")
            return
        
        # 統計結果信息
        sources = list(set([r['metadata'].get('filename', 'unknown') for r in results]))
        categories = list(set([r['metadata'].get('category', 'unknown') for r in results]))
        similarities = [r['similarity'] for r in results]
        semantic_boosts = [r.get('semantic_boost', 0.0) for r in results]
        
        # 統計來自 semantic_analysis.md 的結果
        semantic_analysis_count = sum(1 for r in results if 'semantic_analysis' in r['metadata'].get('filename', ''))
        
        self.logger.info(f"🔍 語義增強搜索完成: {len(results)} 個結果")
        self.logger.info(f"   📚 來源文件: {', '.join(sources)}")
        self.logger.info(f"   📂 涉及類別: {', '.join(categories)}")
        self.logger.info(f"   📊 相似度範圍: {min(similarities):.3f} - {max(similarities):.3f}")
        self.logger.info(f"   🎯 語義加成範圍: {min(semantic_boosts):.3f} - {max(semantic_boosts):.3f}")
        self.logger.info(f"   🧠 語義分析文件結果: {semantic_analysis_count}/{len(results)}")
        
        # 記錄情感分析信息
        if emotional_analysis:
            emotional_intensity = emotional_analysis.get('emotional_intensity', 'mild')
            detected_intent = emotional_analysis.get('detected_intent', 'unknown')
            intimacy_score = emotional_analysis.get('intimacy_score', 0.0)
            
            self.logger.info(f"   💭 情感強度: {emotional_intensity}")
            self.logger.info(f"   🎯 檢測意圖: {detected_intent}")
            self.logger.info(f"   💝 親密度: {intimacy_score:.2f}")
    
    def get_semantic_analysis_stats(self) -> Dict[str, Any]:
        """獲取語義分析相關統計信息"""
        try:
            # 獲取所有結果
            all_results = self.collection.get(include=['metadatas'])
            
            # 統計語義分析相關信息
            semantic_analysis_count = 0
            content_category_count = 0
            total_count = len(all_results['metadatas']) if all_results['metadatas'] else 0
            
            if all_results['metadatas']:
                for metadata in all_results['metadatas']:
                    filename = metadata.get('filename', '')
                    category = metadata.get('category', '')
                    
                    if 'semantic_analysis' in filename:
                        semantic_analysis_count += 1
                    
                    if category == 'content':
                        content_category_count += 1
            
            return {
                'total_documents': total_count,
                'semantic_analysis_chunks': semantic_analysis_count,
                'content_category_chunks': content_category_count,
                'semantic_search_mapping_count': len(self.semantic_search_mapping),
                'enhanced_categories': list(self.knowledge_categories.keys()),
                'semantic_features_enabled': True
            }
            
        except Exception as e:
            self.logger.error(f"獲取語義分析統計失敗: {e}")
            return {
                'total_documents': 0,
                'semantic_analysis_chunks': 0,
                'semantic_features_enabled': False
            }
    
    async def _initialize_semantic_vectors(self):
        """初始化預計算的語義向量"""
        if self.vector_cache_initialized:
            return
            
        try:
            # 預計算常用語義向量
            semantic_texts = {
                'comfort': '溫柔安慰 體貼關懷 沒關係 我在這裡 陪伴支持',
                'companionship': '一起陪伴 溫暖互動 不孤獨 在一起 陪你',
                'rushia_personality': '露西亞 溫柔 可愛 死靈法師 不服輸 黏人',
                'positive_interaction': '開心 快樂 有趣 好玩 愉快 歡樂',
                'intimate_closeness': '親密 親愛的 撒嬌 甜蜜 溫暖 關心',
                'character_traits': '性格 特徵 個性 特質 行為 風格',
                'factual_info': '資料 信息 事實 數據 基本 介紹'
            }
            
            for key, text in semantic_texts.items():
                vector = self.embedding_model.encode([text])[0]
                if hasattr(vector, 'cpu'):
                    vector = vector.cpu().numpy()
                self.semantic_vectors[key] = vector
            
            self.vector_cache_initialized = True
            self.logger.info("✅ 語義向量緩存初始化完成")
            
        except Exception as e:
            self.logger.error(f"❌ 語義向量初始化失敗: {e}")
    
    def _adjust_query_vector(self, query_vector, emotional_analysis):
        """根據情感分析調整查詢向量"""
        if not emotional_analysis or not self.vector_cache_initialized:
            return query_vector
        
        try:
            # 轉換為numpy數組
            if hasattr(query_vector, 'cpu'):
                adjusted_vector = query_vector.cpu().numpy()
            else:
                adjusted_vector = np.array(query_vector)
            
            # 根據情感強度混合情感向量
            emotional_intensity = emotional_analysis.get('emotional_intensity', 'mild')
            if emotional_intensity == 'very_strong' and 'comfort' in self.semantic_vectors:
                comfort_vector = self.semantic_vectors['comfort']
                adjusted_vector = 0.7 * adjusted_vector + 0.3 * comfort_vector
            
            # 根據意圖混合相應向量
            detected_intent = emotional_analysis.get('detected_intent', '')
            if detected_intent == 'companionship_request' and 'companionship' in self.semantic_vectors:
                companion_vector = self.semantic_vectors['companionship']
                adjusted_vector = 0.8 * adjusted_vector + 0.2 * companion_vector
            elif detected_intent == 'intimate_expression' and 'intimate_closeness' in self.semantic_vectors:
                intimate_vector = self.semantic_vectors['intimate_closeness']
                adjusted_vector = 0.8 * adjusted_vector + 0.2 * intimate_vector
            
            # 總是混合一點露西亞人格向量，確保角色一致性
            if 'rushia_personality' in self.semantic_vectors:
                personality_vector = self.semantic_vectors['rushia_personality']
                adjusted_vector = 0.9 * adjusted_vector + 0.1 * personality_vector
            
            return adjusted_vector
            
        except Exception as e:
            self.logger.error(f"調整查詢向量失敗: {e}")
            return query_vector
    
    async def _multi_vector_search(self, adjusted_vector, emotional_analysis, top_k):
        """多向量搜索策略"""
        try:
            # 主搜索：使用調整後的向量
            main_results = await self._vector_search_with_vector(adjusted_vector, top_k=top_k * 2)
            
            # 補充搜索：確保角色一致性
            if 'rushia_personality' in self.semantic_vectors:
                char_vector = self.semantic_vectors['rushia_personality']
                char_results = await self._vector_search_with_vector(char_vector, top_k=3)
                
                # 合併結果，避免重複
                combined_results = self._merge_search_results(main_results, char_results)
            else:
                combined_results = main_results
            
            return combined_results
            
        except Exception as e:
            self.logger.error(f"多向量搜索失敗: {e}")
            return []
    
    async def _vector_search_with_vector(self, query_vector, top_k=5):
        """使用給定向量進行搜索"""
        try:
            # 確保向量格式正確
            if hasattr(query_vector, 'tolist'):
                query_embedding = query_vector.tolist()
            else:
                query_embedding = list(query_vector)
            
            # 執行向量搜索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # 格式化結果
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    similarity = 1 - results['distances'][0][i]
                    metadata = results['metadatas'][0][i]
                    
                    if similarity >= self.rag_config['retrieval']['similarity_threshold']:
                        weight = metadata.get('weight', 1.0)
                        weighted_score = similarity * weight
                        
                        search_results.append({
                            'content': results['documents'][0][i],
                            'metadata': metadata,
                            'similarity': similarity,
                            'weighted_score': weighted_score
                        })
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"向量搜索失敗: {e}")
            return []
    
    def _merge_search_results(self, main_results, supplementary_results):
        """合併搜索結果，避免重複"""
        seen_ids = set()
        merged_results = []
        
        # 添加主要結果
        for result in main_results:
            content_hash = hash(result['content'][:100])  # 使用內容前100字符作為唯一標識
            if content_hash not in seen_ids:
                merged_results.append(result)
                seen_ids.add(content_hash)
        
        # 添加補充結果（避免重複）
        for result in supplementary_results:
            content_hash = hash(result['content'][:100])
            if content_hash not in seen_ids:
                result['supplementary'] = True  # 標記為補充結果
                merged_results.append(result)
                seen_ids.add(content_hash)
        
        # 按權重分數排序
        merged_results.sort(key=lambda x: x['weighted_score'], reverse=True)
        return merged_results
    
    def _ensure_result_diversity(self, results):
        """確保結果多樣性"""
        if not results:
            return results
        
        file_counts = {}
        diverse_results = []
        
        # 特殊處理semantic_analysis.md
        semantic_analysis_count = 0
        
        for result in results:
            filename = result['metadata'].get('filename', 'unknown')
            
            # 限制semantic_analysis.md的結果數量
            if 'semantic_analysis' in filename:
                if semantic_analysis_count >= self.diversity_config['semantic_analysis_limit']:
                    continue
                semantic_analysis_count += 1
            
            # 限制每個文件的結果數量
            current_count = file_counts.get(filename, 0)
            if current_count >= self.diversity_config['max_per_file']:
                continue
            
            diverse_results.append(result)
            file_counts[filename] = current_count + 1
        
        # 記錄多樣性信息
        unique_files = len(file_counts)
        self.logger.info(f"🎯 結果多樣性: {unique_files} 個不同文件, semantic_analysis: {semantic_analysis_count}")
        
        return diverse_results