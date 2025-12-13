import base64
import io
import json
import random
import re
import asyncio
from pathlib import Path
from typing import List

import aiohttp
from PIL import Image as PILImage
import astrbot.core.message.components as Comp
from astrbot.api import logger
from astrbot.api.event import MessageChain, AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.message.components import At, Image, Reply, Video

@register("astrbot_plugin_sora", "gaogxb", "使用newAPI生成视频。指令 文生视频 <提示词> 或 图生视频 <提示词> + 图片", "1.0")
class ApiVideoPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.api_key = config.get("api_key", "")
        self.api_url = config.get("api_url", "https://grsai.dakka.com.cn/v1/video/sora-video")
        self.model = config.get("model", "sora-2")
        self.session = aiohttp.ClientSession()
        
    async def terminate(self):
        if self.session and not self.session.closed:
            await self.session.close()

    @filter.regex(r"^(文生视频)", priority=3)
    async def handle_text_to_video(self, event: AstrMessageEvent):
        user_prompt = re.sub(
            r"^(文生视频)\s*", "", event.message_obj.message_str, count=1
        ).strip()

        if not self.api_key:
            yield event.plain_result("错误：请先在配置文件中设置api_key")
            return
        
        if not user_prompt:
            yield event.plain_result("请输入视频生成的提示词。用法: 文生视频 <提示词>")
            return

        yield event.plain_result("收到文生视频请求，生成中请稍候...")
        payload = self._build_payload(user_prompt)
        
        await self._generate_and_send_video(event, payload)

    @filter.regex(r"^(图生视频)", priority=3)
    async def handle_image_to_video(self, event: AstrMessageEvent):
        user_prompt = re.sub(
            r"^(图生视频)\s*", "", event.message_obj.message_str, count=1
        ).strip()
        
        if not self.api_key:
            yield event.plain_result("错误：请先在配置文件中设置api_key")
            return
            
        image_list = await self.get_images(event)
        if not image_list:
            # 修改点 2: 更新帮助文本
            yield event.plain_result("请提供一张或多张图片来生成视频。用法: 图生视频 <提示词> + 图片")
            return

        if not user_prompt:
            user_prompt = "让画面动起来"
        
        # 改进用户反馈，告知收到了多少张图片
        yield event.plain_result(f"收到图生视频请求，共计 {len(image_list)} 张图片，生成中请稍候...")
        
        # sora2 API 只支持一张参考图，使用第一张图片
        image_bytes = image_list[0]
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        # sora2 支持 base64 格式，格式为: data:image/png;base64,{base64_string}
        image_url = f"data:image/png;base64,{base64_image}"
        
        payload = self._build_payload(user_prompt, image_url)
        await self._generate_and_send_video(event, payload)

    def _build_payload(self, prompt: str, image_url: str = None) -> dict:
        """构建 sora2 API 请求参数"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "aspectRatio": self.config.get("aspectRatio", "16:9"),
            "duration": self.config.get("duration", 10),
            "size": self.config.get("size", "small"),
            "shutProgress": self.config.get("shutProgress", False)
        }
        
        # 如果有参考图，添加到 url 字段
        if image_url:
            payload["url"] = image_url
        
        return payload

    async def _generate_and_send_video(self, event: AstrMessageEvent, payload: dict):
        """使用 sora2 API 生成视频（流式响应）"""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        try:
            async with self.session.post(self.api_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    # sora2 使用流式响应，每行是一个 JSON 对象（不是 SSE 格式，没有 data: 前缀）
                    video_url = None
                    last_progress = 0
                    task_id = None
                    
                    buffer = ""
                    async for chunk in response.content.iter_chunked(1024):
                        buffer += chunk.decode('utf-8', errors='ignore')
                        # 按行处理缓冲区中的数据
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line_str = line.strip()
                            if not line_str:
                                continue
                            
                            try:
                                # 每行是直接的 JSON 对象
                                data = json.loads(line_str)
                                logger.debug(f"解析到流式数据: {json.dumps(data, ensure_ascii=False)[:200]}")
                                
                                # 获取任务 ID
                                if "id" in data and not task_id:
                                    task_id = data["id"]
                                    logger.info(f"任务 ID: {task_id}")
                                
                                # 检查状态
                                status = data.get("status", "")
                                progress = data.get("progress", 0)
                                
                                # 更新进度（每 10% 更新一次，避免消息过多）
                                if progress > last_progress + 10:
                                    last_progress = progress
                                    logger.info(f"视频生成进度: {progress}%")
                                
                                # 检查是否成功
                                if status == "succeeded":
                                    results = data.get("results", [])
                                    if results and len(results) > 0:
                                        video_url = results[0].get("url")
                                        if video_url:
                                            logger.info(f"成功获取视频链接: {video_url}")
                                            # 发送视频
                                            await event.send(event.chain_result([Video.fromURL(url=video_url)]))
                                            logger.info("已成功向框架提交视频URL。")
                                            return
                                        else:
                                            logger.warning("results 中没有 url 字段")
                                    else:
                                        logger.warning("results 为空")
                                
                                # 检查是否失败
                                elif status == "failed":
                                    failure_reason = data.get("failure_reason", "")
                                    error = data.get("error", "")
                                    error_msg = f"视频生成失败: {failure_reason}"
                                    if error:
                                        error_msg += f" - {error}"
                                    logger.error(error_msg)
                                    await self.context.send_message(
                                        event.unified_msg_origin, 
                                        MessageChain().message(error_msg)
                                    )
                                    return
                                
                            except json.JSONDecodeError as e:
                                logger.debug(f"JSON解析失败: {e}, 行内容: {line_str[:100]}")
                                continue
                            except Exception as e:
                                logger.debug(f"解析流式数据行时出错: {e}, 行内容: {line_str[:100]}")
                                continue
                    
                    # 处理缓冲区中剩余的数据
                    if buffer.strip():
                        try:
                            data = json.loads(buffer.strip())
                            status = data.get("status", "")
                            if status == "succeeded":
                                results = data.get("results", [])
                                if results and len(results) > 0:
                                    video_url = results[0].get("url")
                                    if video_url:
                                        logger.info(f"成功获取视频链接: {video_url}")
                                        await event.send(event.chain_result([Video.fromURL(url=video_url)]))
                                        logger.info("已成功向框架提交视频URL。")
                                        return
                        except:
                            pass
                    
                    # 如果流式处理完成但没有获取到视频链接
                    if not video_url:
                        error_msg = "未能从流式响应中获取视频链接，请检查日志"
                        logger.error(error_msg)
                        await self.context.send_message(
                            event.unified_msg_origin, 
                            MessageChain().message(error_msg)
                        )
                else:
                    error_text = await response.text()
                    error_msg = f"API请求失败，状态码: {response.status}, 响应: {error_text}"
                    logger.error(error_msg)
                    await self.context.send_message(
                        event.unified_msg_origin, 
                        MessageChain().message(error_msg)
                    )

        except Exception as e:
            logger.error(f"视频生成过程中发生严重错误: {e}", exc_info=True)
            await self.context.send_message(
                event.unified_msg_origin, 
                MessageChain().message(f"生成失败: {str(e)}")
            )

        
    async def get_images(self, event: AstrMessageEvent) -> list[bytes]:
        images = []
        for s in event.message_obj.message:
            if isinstance(s, Comp.Reply) and s.chain:
                for seg in s.chain:
                    if isinstance(seg, Comp.Image):
                        if seg.url and (img := await self._load_bytes(seg.url)):
                            images.append(img)
                        elif seg.file and (img := await self._load_bytes(seg.file)):
                            images.append(img)
        if images: return images
        for seg in event.message_obj.message:
            if isinstance(seg, Comp.Image):
                if seg.url and (img := await self._load_bytes(seg.url)):
                    images.append(img)
                elif seg.file and (img := await self._load_bytes(seg.file)):
                    images.append(img)
        if images: return images
        for seg in event.message_obj.message:
            if isinstance(seg, Comp.At):
                if avatar := await self._get_avatar(str(seg.qq)):
                    images.append(avatar)
        return images

    async def _download_image(self, url: str) -> bytes | None:
        try:
            async with self.session.get(url) as resp:
                resp.raise_for_status()
                return await resp.read()
        except Exception as e:
            logger.error(f"图片下载失败: {e}")
            return None

    async def _get_avatar(self, user_id: str) -> bytes | None:
        if not user_id.isdigit(): user_id = "".join(random.choices("0123456789", k=9))
        avatar_url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
        try:
            async with self.session.get(avatar_url, timeout=10) as resp:
                resp.raise_for_status()
                return await resp.read()
        except Exception as e:
            logger.error(f"下载头像失败: {e}")
            return None

    def _extract_first_frame_sync(self, raw: bytes) -> bytes:
        img_io = io.BytesIO(raw)
        img = PILImage.open(img_io)
        if img.format != "GIF": return raw
        logger.info("检测到GIF, 将抽取 GIF 的第一帧来生图")
        first_frame = img.convert("RGBA")
        out_io = io.BytesIO()
        first_frame.save(out_io, format="PNG")
        return out_io.getvalue()

    async def _load_bytes(self, src: str) -> bytes | None:
        raw: bytes | None = None
        loop = asyncio.get_running_loop()
        path = Path(src)
        if path.is_file():
            raw = await loop.run_in_executor(None, path.read_bytes)
        elif src.startswith("http"):
            raw = await self._download_image(src)
        elif src.startswith("base64://"):
            raw = await loop.run_in_executor(None, base64.b64decode, src[9:])
        if not raw: return None
        return await loop.run_in_executor(None, self._extract_first_frame_sync, raw)
