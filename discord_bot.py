#!/usr/bin/env python3
"""
VTuber AI Discord Bot - 
使用核心服務層處理所有AI邏輯，指令註冊
"""
import discord
from discord.ext import commands
import asyncio
import os
import sys
import logging
from pathlib import Path
import yaml
from datetime import datetime, timedelta
from dotenv import load_dotenv

from src.core_service import VTuberCoreService
from src.utils.logger import setup_logger


class VTuberDiscordBot(commands.Bot):
    """輕量級Discord Bot - 修復版本"""
    
    def __init__(self, config: dict, authorized_users: list, authorized_guilds: list = None, bot_owner_id: int = None):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.dm_messages = True
        
        # 禁用內建的help指令以避免衝突
        super().__init__(command_prefix='/', intents=intents, owner_id=bot_owner_id, help_command=None)
        
        self.config = config
        self.authorized_users = set(authorized_users)
        self.authorized_guilds = set(authorized_guilds) if authorized_guilds else set()
        self.bot_owner_id = bot_owner_id
        
        # 核心服務
        self.core_service = VTuberCoreService(config)
        
        # 簡單的並發控制
        self.max_concurrent_requests = 3  # 🔥 降低併發數量，避免阻塞
        self.request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self.current_requests = 0
        
        # 速率限制
        self.user_request_times = {}  # user_id -> list of request times
        self.max_requests_per_minute = 5  # 🔥 降低速率限制，避免過載
        
        self.logger = logging.getLogger(__name__)
        
        # 設置指令 - 在初始化後立即註冊
        self.setup_commands()
    
    def setup_commands(self):
        """設置所有指令 - 使用動態註冊避免 CommandNotFound 錯誤"""
        
        # ==================== 基本指令 ====================
        
        @self.command(name='help')
        async def help_cmd(ctx):
            """幫助信息"""
            # 檢查核心服務是否已初始化
            if not self._check_core_service_ready():
                await self._send_not_ready_message(ctx)
                return
                
            try:
                stats = self.core_service.get_stats()
                character_name = stats.get('character_name', 'AI助手')
            except Exception as e:
                self.logger.error(f"獲取統計信息失敗: {e}")
                character_name = 'AI助手'
            
            embed = discord.Embed(
                title=f"🎭 {character_name} Discord Bot 幫助",
                description="與AI助手聊天的指令說明",
                color=0x9932CC
            )
            
            embed.add_field(
                name="💬 基本對話",
                value="• 直接@我或私訊即可開始對話\n• 支援中文、英文等多語言",
                inline=False
            )
            
            embed.add_field(
                name="📚 知識庫管理",
                value="• `/add` - 上傳文檔到知識庫\n• `/search <關鍵詞>` - 搜索知識庫\n• `/clear_kb` - 清空知識庫",
                inline=False
            )
            
            embed.add_field(
                name="🔧 系統指令",
                value="• `/stats` - 顯示系統統計\n• `/model` - 顯示模型信息\n• `/memory` - 查看對話記憶\n• `/clear_memory` - 清除對話記憶\n• `/cancel` - 取消當前生成",
                inline=False
            )
            
            embed.add_field(
                name="⚙️ 控制指令",
                value="• `/rag_on` - 啟用RAG檢索\n• `/rag_off` - 禁用RAG檢索\n• `/s2t_on` - 啟用簡繁轉換\n• `/s2t_off` - 禁用簡繁轉換\n• `/typing_on` - 啟用打字模擬\n• `/typing_off` - 禁用打字模擬",
                inline=False
            )
            
            if self.is_owner(ctx.author.id):
                embed.add_field(
                    name="👑 管理員指令",
                    value="• `/shutdown` - 關閉Bot\n• `/reload` - 重載配置\n• `/status` - 詳細狀態",
                    inline=False
                )
            
            embed.set_footer(text=f"當前並發請求: {self.current_requests}/{self.max_concurrent_requests}")
            await ctx.send(embed=embed)
        
        @self.command(name='stats')
        async def stats_cmd(ctx):
            """系統統計"""
            if not self.is_authorized(ctx.author.id, getattr(ctx.guild, 'id', None)):
                await ctx.send("❌ 權限不足")
                return
            
            if not self._check_core_service_ready():
                await self._send_not_ready_message(ctx)
                return
                
            try:
                stats = self.core_service.get_stats()
                if not stats.get('success'):
                    await ctx.send(f"❌ 獲取統計失敗: {stats.get('error')}")
                    return
            except Exception as e:
                await ctx.send(f"❌ 獲取統計失敗: {str(e)}")
                return
            
            embed = discord.Embed(title="📊 系統統計", color=0x00FF00)
            embed.add_field(name="📚 知識庫文檔", value=f"{stats['total_documents']} 個", inline=True)
            embed.add_field(name="🔍 RAG狀態", value="✅ 啟用" if stats['rag_enabled'] else "❌ 禁用", inline=True)
            embed.add_field(name="👥 活躍用戶", value=f"{stats['active_users']} 人", inline=True)
            embed.add_field(name="🎭 角色", value=stats['character_name'], inline=True)
            embed.add_field(name="💫 性格", value=stats['character_personality'], inline=True)
            embed.add_field(name="🤖 並發請求", value=f"{self.current_requests}/{self.max_concurrent_requests}", inline=True)
            
            await ctx.send(embed=embed)
        
        @self.command(name='model')
        async def model_cmd(ctx):
            """模型信息"""
            if not self.is_authorized(ctx.author.id, getattr(ctx.guild, 'id', None)):
                await ctx.send("❌ 權限不足")
                return
            
            if not self._check_core_service_ready():
                await self._send_not_ready_message(ctx)
                return
                
            try:
                model_info = self.core_service.get_model_info()
                if "error" in model_info:
                    await ctx.send(f"❌ 獲取模型信息失敗: {model_info['error']}")
                    return
            except Exception as e:
                await ctx.send(f"❌ 獲取模型信息失敗: {str(e)}")
                return
            
            embed = discord.Embed(title="🤖 模型信息", color=0x00BFFF)
            
            # 主模型信息
            llm_info = model_info.get('llm_model', {})
            embed.add_field(
                name="📦 主模型 (LLM)",
                value=f"類型: {llm_info.get('model_type', 'Unknown')}\n"
                      f"量化: {llm_info.get('quantization', 'Unknown')}\n"
                      f"設備: {llm_info.get('device', 'Unknown')}",
                inline=True
            )
            
            # 嵌入模型信息
            emb_info = model_info.get('embedding_model', {})
            embed.add_field(
                name="🔍 嵌入模型",
                value=f"類型: {emb_info.get('model_type', 'Unknown')}\n"
                      f"量化: {emb_info.get('quantization', 'Unknown')}\n"
                      f"設備: {emb_info.get('device', 'Unknown')}",
                inline=True
            )
            
            # GPU信息
            if 'gpu_total' in llm_info:
                embed.add_field(
                    name="🎮 GPU 記憶體",
                    value=f"總計: {llm_info.get('gpu_total', 'Unknown')}\n"
                          f"已分配: {llm_info.get('gpu_allocated', 'Unknown')}\n"
                          f"已緩存: {llm_info.get('gpu_cached', 'Unknown')}",
                    inline=True
                )
            
            await ctx.send(embed=embed)
        
        # ==================== 知識庫管理指令 ====================
        
        @self.command(name='add')
        async def add_cmd(ctx):
            """添加文檔"""
            if not self.is_authorized(ctx.author.id, getattr(ctx.guild, 'id', None)):
                await ctx.send("❌ 權限不足")
                return
            
            if not ctx.message.attachments:
                await ctx.send("❌ 請附加文件\n支援格式: .txt, .pdf, .docx")
                return
            
            for attachment in ctx.message.attachments:
                try:
                    # 檢查文件大小（限制10MB）
                    if attachment.size > 10 * 1024 * 1024:
                        await ctx.send(f"❌ 文件 `{attachment.filename}` 太大（限制10MB）")
                        continue
                    
                    # 下載文件
                    temp_dir = Path("temp")
                    temp_dir.mkdir(exist_ok=True)
                    file_path = temp_dir / attachment.filename
                    
                    await attachment.save(file_path)
                    
                    # 添加到知識庫
                    result = await self.core_service.add_document(str(file_path))
                    
                    if result['success']:
                        await ctx.send(f"✅ 成功添加文件: `{attachment.filename}`")
                    else:
                        await ctx.send(f"❌ 添加文件失敗: {result.get('error')}")
                    
                    # 清理臨時文件
                    try:
                        file_path.unlink()
                    except:
                        pass
                        
                except Exception as e:
                    await ctx.send(f"❌ 處理文件 `{attachment.filename}` 失敗: {str(e)}")
        
        @self.command(name='search')
        async def search_cmd(ctx, *, query: str = None):
            """搜索知識庫"""
            if not self.is_authorized(ctx.author.id, getattr(ctx.guild, 'id', None)):
                await ctx.send("❌ 權限不足")
                return
            
            if not query:
                await ctx.send("❌ 請提供搜索關鍵詞\n用法: `/search 你的關鍵詞`")
                return
            
            result = await self.core_service.search_knowledge_base(query, top_k=3)
            
            if not result['success']:
                await ctx.send(f"❌ 搜索失敗: {result.get('error')}")
                return
            
            results = result['results']
            if not results:
                await ctx.send("📭 未找到相關內容")
                return
            
            embed = discord.Embed(
                title=f"🔍 搜索結果: {query}",
                description=f"找到 {result['count']} 條相關內容",
                color=0xFFD700
            )
            
            for i, item in enumerate(results[:3], 1):  # 只顯示前3個結果
                source = item['metadata'].get('filename', '未知來源')
                content = item['content'][:150] + "..." if len(item['content']) > 150 else item['content']
                similarity = item['similarity']
                
                embed.add_field(
                    name=f"[{i}] {source} (相關度: {similarity:.2f})",
                    value=content,
                    inline=False
                )
            
            await ctx.send(embed=embed)
        
        # ==================== 記憶管理指令 ====================
        
        @self.command(name='memory')
        async def memory_cmd(ctx):
            """查看對話記憶"""
            if not self.is_authorized(ctx.author.id, getattr(ctx.guild, 'id', None)):
                await ctx.send("❌ 權限不足")
                return
            
            user_id = str(ctx.author.id)
            memory_status = self.core_service.get_user_memory_status(user_id)
            
            if not memory_status.get('success'):
                await ctx.send(f"❌ 獲取記憶狀態失敗: {memory_status.get('error')}")
                return
            
            embed = discord.Embed(title="💭 對話記憶狀態", color=0x9932CC)
            embed.add_field(
                name="記憶輪數",
                value=f"{memory_status['memory_count']}/{memory_status['max_length']}",
                inline=True
            )
            
            if memory_status.get('last_active'):
                embed.add_field(
                    name="最後活躍",
                    value=memory_status['last_active'][:19],  # 去掉毫秒
                    inline=True
                )
            
            history = memory_status.get('history', [])
            if history:
                stats = self.core_service.get_stats()
                character_name = stats.get('character_name', 'AI助手')
                
                for i, (user_msg, bot_response) in enumerate(history, 1):
                    user_short = user_msg[:100] + "..." if len(user_msg) > 100 else user_msg
                    bot_short = bot_response[:100] + "..." if len(bot_response) > 100 else bot_response
                    
                    embed.add_field(
                        name=f"對話 {i}",
                        value=f"**你:** {user_short}\n**{character_name}:** {bot_short}",
                        inline=False
                    )
            else:
                embed.add_field(name="記憶內容", value="暫無對話記憶", inline=False)
            
            await ctx.send(embed=embed)
        
        @self.command(name='clear_memory')
        async def clear_memory_cmd(ctx):
            """清除對話記憶"""
            if not self.is_authorized(ctx.author.id, getattr(ctx.guild, 'id', None)):
                await ctx.send("❌ 權限不足")
                return
            
            user_id = str(ctx.author.id)
            result = self.core_service.clear_user_memory(user_id)
            
            if result['success']:
                await ctx.send(f"✅ {result['message']}")
            else:
                await ctx.send(f"❌ {result.get('error', '清除記憶失敗')}")
        
        @self.command(name='clear_kb')
        async def clear_kb_cmd(ctx):
            """清空知識庫"""
            if not self.is_owner(ctx.author.id):
                await ctx.send("❌ 只有Bot擁有者可以執行此操作")
                return
            
            # 確認操作
            await ctx.send("⚠️ 確定要清空整個知識庫嗎？回覆 `yes` 確認")
            
            def check(m):
                return m.author == ctx.author and m.channel == ctx.channel
            
            try:
                msg = await self.wait_for('message', check=check, timeout=30.0)
                if msg.content.lower() == 'yes':
                    result = await self.core_service.clear_knowledge_base()
                    if result['success']:
                        await ctx.send(f"✅ {result['message']}")
                    else:
                        await ctx.send(f"❌ {result.get('error')}")
                else:
                    await ctx.send("操作已取消")
            except asyncio.TimeoutError:
                await ctx.send("操作超時，已取消")
        
        # ==================== 功能控制指令 ====================
        
        @self.command(name='rag_on')
        async def rag_on_cmd(ctx):
            """啟用RAG"""
            if not self.is_authorized(ctx.author.id, getattr(ctx.guild, 'id', None)):
                await ctx.send("❌ 權限不足")
                return
            
            result = self.core_service.toggle_rag(True)
            await ctx.send(f"✅ {result['message']}")
        
        @self.command(name='rag_off')
        async def rag_off_cmd(ctx):
            """禁用RAG"""
            if not self.is_authorized(ctx.author.id, getattr(ctx.guild, 'id', None)):
                await ctx.send("❌ 權限不足")
                return
            
            result = self.core_service.toggle_rag(False)
            await ctx.send(f"⚠️ {result['message']}")
        
        @self.command(name='s2t_on')
        async def s2t_on_cmd(ctx):
            """啟用簡繁轉換"""
            if not self.is_authorized(ctx.author.id, getattr(ctx.guild, 'id', None)):
                await ctx.send("❌ 權限不足")
                return
            
            result = self.core_service.toggle_traditional_chinese(True)
            if result['success']:
                await ctx.send(f"✅ {result['message']}")
            else:
                await ctx.send(f"⚠️ {result['message']}")
        
        @self.command(name='s2t_off')
        async def s2t_off_cmd(ctx):
            """禁用簡繁轉換"""
            if not self.is_authorized(ctx.author.id, getattr(ctx.guild, 'id', None)):
                await ctx.send("❌ 權限不足")
                return
            
            result = self.core_service.toggle_traditional_chinese(False)
            if result['success']:
                await ctx.send(f"⚠️ {result['message']}")
            else:
                await ctx.send(f"❌ {result.get('error')}")
        
        @self.command(name='typing_on')
        async def typing_on_cmd(ctx):
            """啟用打字模擬"""
            if not self.is_authorized(ctx.author.id, getattr(ctx.guild, 'id', None)):
                await ctx.send("❌ 權限不足")
                return
            
            result = self.core_service.toggle_typing_simulation(True)
            await ctx.send("✅ 已啟用人性化打字模擬，回應將逐行顯示")
        
        @self.command(name='typing_off')
        async def typing_off_cmd(ctx):
            """禁用打字模擬"""
            if not self.is_authorized(ctx.author.id, getattr(ctx.guild, 'id', None)):
                await ctx.send("❌ 權限不足")
                return
            
            result = self.core_service.toggle_typing_simulation(False)
            await ctx.send("⚠️ 已禁用打字模擬，將直接顯示完整回應")
        
        # ==================== 管理員指令 ====================
        
        @self.command(name='shutdown')
        async def shutdown_cmd(ctx):
            """關閉Bot"""
            if not self.is_owner(ctx.author.id):
                await ctx.send("❌ 只有Bot擁有者可以執行此操作")
                return
            
            await ctx.send("👋 Bot即將關閉...")
            await self.close()
        
        @self.command(name='status')
        async def status_cmd(ctx):
            """詳細狀態"""
            if not self.is_owner(ctx.author.id):
                await ctx.send("❌ 只有Bot擁有者可以執行此操作")
                return
            
            embed = discord.Embed(title="🔧 Bot詳細狀態", color=0xFF6347)
            embed.add_field(name="網路延遲", value=f"{round(self.latency * 1000)}ms", inline=True)
            embed.add_field(name="伺服器數", value=len(self.guilds), inline=True)
            embed.add_field(name="可見用戶數", value=len(self.users), inline=True)
            embed.add_field(name="當前請求", value=f"{self.current_requests}/{self.max_concurrent_requests}", inline=True)
            embed.add_field(name="速率限制記錄", value=len(self.user_request_times), inline=True)
            embed.add_field(name="打字模擬", value="已啟用" if hasattr(self.core_service, 'typing_simulation_enabled') and self.core_service.typing_simulation_enabled else "已禁用", inline=True)
            
            # 獲取核心服務狀態
            try:
                core_stats = self.core_service.get_stats()
                if core_stats.get('success'):
                    embed.add_field(name="活躍AI會話", value=f"{core_stats['active_users']} 個", inline=True)
                    embed.add_field(name="RAG狀態", value="✅ 啟用" if core_stats['rag_enabled'] else "❌ 禁用", inline=True)
                    embed.add_field(name="知識庫文檔", value=f"{core_stats['total_documents']} 個", inline=True)
            except Exception as e:
                embed.add_field(name="核心服務狀態", value=f"❌ 錯誤: {str(e)}", inline=False)
            
            await ctx.send(embed=embed)
        
        @self.command(name='cancel')
        async def cancel_cmd(ctx):
            """取消當前生成"""
            if not self.is_authorized(ctx.author.id, getattr(ctx.guild, 'id', None)):
                await ctx.send("❌ 權限不足")
                return
            
            # 流式生成的取消比較複雜，這裡暫時提供一個提示
            await ctx.send("ℹ️ 流式生成中的取消功能正在開發中，請等待當前回應完成")
        
        print(f"✅ 已註冊 {len(self.commands)} 個指令")
    
    async def setup_hook(self):
        """Bot啟動設置"""
        print("🚀 正在初始化Discord Bot...")
        
        success = await self.core_service.initialize()
        if not success:
            print("❌ 核心服務初始化失敗")
            await self.close()
            return
        
        print("✅ Discord Bot初始化完成!")
    
    async def on_ready(self):
        """Bot準備就緒"""
        stats = self.core_service.get_stats()
        character_name = stats.get('character_name', 'AI助手')
        
        print(f"🎭 {character_name} Discord Bot 已上線!")
        print(f"📋 Bot ID: {self.user.id}")
        print(f"👥 授權用戶數量: {len(self.authorized_users)}")
        print(f"🏠 授權伺服器數量: {len(self.authorized_guilds)}")
        print(f"📝 已註冊指令數量: {len(self.commands)} (可用 /help 查看)")
        
        activity = discord.Activity(
            type=discord.ActivityType.listening,
            name=f"與{character_name}聊天 | /help"
        )
        await self.change_presence(activity=activity)
    
    async def on_command_error(self, ctx, error):
        """處理指令錯誤"""
        if isinstance(error, commands.CommandNotFound):
            # 不顯示 CommandNotFound 錯誤，保持沉默
            return
        elif isinstance(error, commands.MissingRequiredArgument):
            await ctx.send(f"❌ 缺少必要參數: {error.param}")
        elif isinstance(error, commands.BadArgument):
            await ctx.send(f"❌ 參數格式錯誤: {error}")
        elif isinstance(error, commands.CommandOnCooldown):
            await ctx.send(f"⏰ 指令冷卻中，請等待 {error.retry_after:.1f} 秒")
        else:
            # 記錄其他錯誤
            self.logger.error(f"指令錯誤: {error}")
            await ctx.send("❌ 執行指令時發生錯誤，請稍後重試")
    
    async def on_message(self, message):
        """處理消息 - 修復指令處理"""
        if message.author == self.user:
            return
        
        # 檢查權限（對所有消息進行檢查）
        is_authorized = self.is_authorized(message.author.id, getattr(message.guild, 'id', None))
        
        # 首先處理指令 - 這很重要！
        # 即使沒有權限，也要處理指令以避免 CommandNotFound 錯誤
        if message.content.startswith('/'):
            # 如果沒有權限，在指令內部處理權限檢查
            await self.process_commands(message)
            return  # 處理完指令就返回
        
        # 對於非指令消息，檢查權限
        if not is_authorized:
            return
        
        # 速率限制檢查（僅對聊天功能）
        if not await self.check_rate_limit(message.author.id):
            await message.channel.send("⏰ 請求過於頻繁，請稍後再試")
            return
        
        # 只處理私訊或提及的非指令消息
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mentioned = self.user in message.mentions
        
        if not (is_dm or is_mentioned):
            return
        
        # 處理聊天內容
        content = self.clean_content(message.content)
        if content.strip():
            await self.handle_chat(message, content.strip())
    
    async def handle_chat(self, message, user_input: str):
        """處理聊天 - 使用流式打字模擬"""
        async with self.request_semaphore:
            thinking_msg = None
            response_msg = None
            try:
                self.current_requests += 1
                user_id = str(message.author.id)
                
                # 發送初始思考消息
                thinking_msg = await message.channel.send("🤔 正在思考中...")
                
                # 使用新的流式生成方法
                response_content = ""
                character_name = "AI助手"
                
                async for chunk in self.core_service.generate_response_with_typing(user_id, user_input):
                    if chunk.get('type') == 'thinking':
                        # 更新思考狀態
                        content = chunk.get('content', '🤔 思考中...')
                        try:
                            await thinking_msg.edit(content=content)
                        except:
                            pass
                        
                    elif chunk.get('type') == 'response_start':
                        # 開始回應，刪除思考消息並創建回應消息
                        character_name = chunk.get('character_name', 'AI助手')
                        try:
                            await thinking_msg.delete()
                        except:
                            pass
                        
                        # 創建初始回應消息（使用佔位符避免空消息錯誤）
                        response_msg = await message.channel.send("...")
                        
                    elif chunk.get('type') == 'response_chunk':
                        # 累積回應內容並更新消息
                        content = chunk.get('content', '')
                        response_content += content
                        
                        # 更新回應消息（限制頻率以避免API限制）
                        try:
                            await response_msg.edit(content=response_content[:2000])
                        except:
                            pass
                            
                    elif chunk.get('type') == 'response_complete':
                        # 回應完成
                        break
                        
                    elif chunk.get('type') == 'error':
                        await message.channel.send(f"❌ 處理失敗: {chunk.get('message')}")
                        break
                        
            except Exception as e:
                self.logger.error(f"處理聊天失敗: {e}")
                # 清理思考消息
                if thinking_msg:
                    try:
                        await thinking_msg.delete()
                    except:
                        pass
                await message.channel.send("❌ 處理消息時發生錯誤，請稍後重試")
            finally:
                self.current_requests -= 1
    
    async def send_response(self, message, response: str):
        """發送回應 - 處理長消息分割"""
        if len(response) > 2000:
            chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
            for chunk in chunks:
                await message.channel.send(chunk)
                await asyncio.sleep(0.1)  # 避免速率限制
        else:
            await message.channel.send(response)
    
    async def check_rate_limit(self, user_id: int) -> bool:
        """檢查速率限制"""
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        
        if user_id not in self.user_request_times:
            self.user_request_times[user_id] = []
        
        # 移除超過1分鐘的請求記錄
        self.user_request_times[user_id] = [
            req_time for req_time in self.user_request_times[user_id] 
            if req_time > one_minute_ago
        ]
        
        # 檢查是否超過限制
        if len(self.user_request_times[user_id]) >= self.max_requests_per_minute:
            return False
        
        # 記錄當前請求
        self.user_request_times[user_id].append(now)
        return True
    
    # ==================== 工具方法 ====================
    
    def _check_core_service_ready(self) -> bool:
        """檢查核心服務是否就緒"""
        return (hasattr(self.core_service, '_initialized') and 
                self.core_service._initialized)
    
    async def _send_not_ready_message(self, ctx):
        """發送服務未就緒消息"""
        await ctx.send("🔄 系統正在初始化中，請稍後再試...")
    
    def is_authorized(self, user_id: int, guild_id: int = None) -> bool:
        """檢查權限"""
        # Bot擁有者總是有權限
        if user_id == self.bot_owner_id:
            return True
        
        # 檢查用戶權限
        if user_id in self.authorized_users:
            # 如果沒有限制伺服器，或者在授權伺服器中
            if not self.authorized_guilds or guild_id in self.authorized_guilds:
                return True
        
        return False
    
    def is_owner(self, user_id: int) -> bool:
        """檢查是否為Bot擁有者"""
        return user_id == self.bot_owner_id
    
    def clean_content(self, content: str) -> str:
        """清理消息內容"""
        content = content.replace(f'<@{self.user.id}>', '').strip()
        content = content.replace(f'<@!{self.user.id}>', '').strip()
        return content
    
    def cleanup(self):
        """清理資源"""
        if self.core_service:
            self.core_service.cleanup()


def load_config(config_path: str = "config.yaml") -> dict:
    """載入配置文件"""
    try:
        config_path = Path(config_path)
        project_root = Path(__file__).parent.parent.parent.resolve()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 處理路徑
        config['models']['llm']['model_path'] = str((project_root / config['models']['llm']['model_path']).resolve())
        config['models']['embedding']['model_path'] = str((project_root / config['models']['embedding']['model_path']).resolve())
        config['rag']['vector_db']['persist_directory'] = str((project_root / config['rag']['vector_db']['persist_directory']).resolve())
        config['system']['cache_dir'] = str((project_root / config['system']['cache_dir']).resolve())
        config['system']['log_dir'] = str((project_root / config['system']['log_dir']).resolve())
        
        return config
    except Exception as e:
        print(f"❌ 載入配置失敗: {e}")
        sys.exit(1)


async def main():
    """主函數"""
    try:
        # 載入環境變量
        load_dotenv()
        
        # 載入配置
        config = load_config()
        
        # 設置日誌
        setup_logger(config['system']['log_dir'])
        
        # Discord Bot 配置
        discord_token = os.getenv('DISCORD_BOT_TOKEN')
        if not discord_token:
            print("❌ 請在 .env 文件中設置 DISCORD_BOT_TOKEN")
            sys.exit(1)
        
        # 授權用戶和伺服器
        authorized_users = [
            int(user_id.strip()) for user_id in os.getenv('AUTHORIZED_USERS', '').split(',') 
            if user_id.strip()
        ]
        
        authorized_guilds = [
            int(guild_id.strip()) for guild_id in os.getenv('AUTHORIZED_GUILDS', '').split(',') 
            if guild_id.strip()
        ]
        
        bot_owner_id = int(os.getenv('BOT_OWNER_ID')) if os.getenv('BOT_OWNER_ID') else None
        
        if not authorized_users and not bot_owner_id:
            print("❌ 請在 .env 文件中設置 AUTHORIZED_USERS 或 BOT_OWNER_ID")
            sys.exit(1)
        
        # 檢查模型文件
        llm_path = Path(config['models']['llm']['model_path'])
        embedding_path = Path(config['models']['embedding']['model_path'])
        
        if not llm_path.exists():
            print(f"❌ LLM 模型路徑不存在: {llm_path}")
            sys.exit(1)
            
        if not embedding_path.exists():
            print(f"❌ 嵌入模型路徑不存在: {embedding_path}")
            sys.exit(1)
        
        # 創建並啟動Bot
        bot = VTuberDiscordBot(
            config=config,
            authorized_users=authorized_users,
            authorized_guilds=authorized_guilds,
            bot_owner_id=bot_owner_id
        )
        
        try:
            await bot.start(discord_token)
        except KeyboardInterrupt:
            # 捕獲 Ctrl+C，優雅關閉
            print("\n👋 Bot已停止")
        finally:
            bot.cleanup()
        
    except KeyboardInterrupt:
        print("\n👋 Bot已停止")
    except Exception as e:
        print(f"❌ Bot運行失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # 最終捕獲，確保不顯示 traceback
        pass
