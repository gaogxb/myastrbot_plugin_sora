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

@register("myastrbot_plugin_sora", "gaogxb", "使用newAPI生成视频。指令 文生视频 <提示词> 或 图生视频 <提示词> + 图片", "1.0.4")
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
        # 处理 shutProgress：如果是字符串 "true" 或 "false"，转换为布尔值
        shut_progress = self.config.get("shutProgress", False)
        if isinstance(shut_progress, str):
            shut_progress = shut_progress.lower() == "true"
        
        payload = {
           "model": self.model,
            "prompt": prompt,
            "aspectRatio": self.config.get("aspectRatio", "16:9"),
            "duration": self.config.get("duration", 10),
            "size": self.config.get("size", "small"),
            "shutProgress": shut_progress
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
            logger.info(f"发送请求到: {self.api_url}")
            logger.debug(f"请求参数: {json.dumps(payload, ensure_ascii=False)}")
            # 设置超时时间（视频生成可能需要较长时间，设置为 5 分钟）
            timeout = aiohttp.ClientTimeout(total=300)
            async with self.session.post(self.api_url, headers=headers, json=payload, timeout=timeout) as response:
                logger.info(f"响应状态码: {response.status}")
                content_type = response.headers.get('Content-Type', '').lower()
                logger.info(f"响应 Content-Type: {content_type}")
                
                if response.status == 200:
                    # 检查是否为流式响应
                    is_stream = 'text/event-stream' in content_type or 'stream' in content_type or 'application/x-ndjson' in content_type
                    
                    # 如果不是流式响应，尝试直接读取 JSON
                    if not is_stream:
                        logger.info("检测到非流式响应，尝试直接解析 JSON")
                        try:
                            data = await response.json()
                            logger.info(f"接收到 JSON 数据: {json.dumps(data, ensure_ascii=False)[:300]}")
                            
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
                        except Exception as e:
                            logger.error(f"解析 JSON 响应失败: {e}")
                            # 继续尝试流式处理
                    
                    # 流式响应处理
                    # sora2 使用流式响应，每行是一个 JSON 对象（不是 SSE 格式，没有 data: 前缀）
                    video_url = None
                    last_progress = 0
                    task_id = None
                    received_lines = []  # 用于调试
                    
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
                                # 记录原始行（前10行用于调试）
                                if len(received_lines) < 10:
                                    received_lines.append(line_str[:200])
                                
                                # 处理 SSE 格式（data: 开头）或直接 JSON
                                json_str = line_str
                                if line_str.startswith('data: '):
                                    json_str = line_str[6:]  # 去掉 'data: ' 前缀
                                    if json_str == '[DONE]':
                                        logger.info("收到流式响应结束标记 [DONE]")
                                        continue
                                
                                # 解析 JSON 对象
                                data = json.loads(json_str)
                                logger.info(f"解析到流式数据: {json.dumps(data, ensure_ascii=False)[:300]}")
                                
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
                            buffer_str = buffer.strip()
                            # 处理 data: 前缀（SSE 格式）
                            if buffer_str.startswith('data: '):
                                buffer_str = buffer_str[6:]
                            if buffer_str and buffer_str != '[DONE]':
                                data = json.loads(buffer_str)
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
                        except Exception as e:
                            logger.debug(f"处理缓冲区剩余数据失败: {e}")
                            pass
                        
                    # 如果流式处理完成但没有获取到视频链接
                    if not video_url:
                        error_msg = "未能从流式响应中获取视频链接"
                        logger.error(error_msg)
                        if received_lines:
                            logger.error(f"接收到的原始数据示例（前{min(10, len(received_lines))}行）: {received_lines}")
                        else:
                            logger.error("未接收到任何数据行")
                        await self.context.send_message(
                            event.unified_msg_origin, 
                            MessageChain().message(f"{error_msg}，请检查日志或联系管理员")
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



async def test_sora_api(api_url: str, api_key: str, model: str = "sora-2"):
    """
    测试 sora2 API 请求函数
    参数:
        api_url: API 地址
        api_key: API 密钥
        model: 模型名称，默认 "sora-2"
    """
    print("=" * 60)
    print("开始测试 sora2 API 请求")
    print("=" * 60)
    print(f"API URL: {api_url}")
    print(f"API Key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else '***'}")
    print(f"Model: {model}")
    print("-" * 60)
    
    # 构建测试请求参数
    payload = {
        "model": model,
        "prompt": "A cute cat playing on the grass",
        "aspectRatio": "16:9",
        "duration": 10,
        "size": "small",
        "shutProgress": False
    }
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    print("请求参数:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("-" * 60)
    print("请求头:")
    print(json.dumps({k: v if k != 'Authorization' else f'Bearer {api_key[:10]}...' for k, v in headers.items()}, indent=2, ensure_ascii=False))
    print("-" * 60)
    
    try:
        async with aiohttp.ClientSession() as session:
            print("正在发送请求...")
            async with session.post(api_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                print(f"\n响应状态码: {response.status}")
                print(f"响应状态文本: {response.reason}")
                print("-" * 60)
                print("响应头:")
                for key, value in response.headers.items():
                    print(f"  {key}: {value}")
                print("-" * 60)
                
                # 读取响应内容
                content_type = response.headers.get('Content-Type', '').lower()
                print(f"Content-Type: {content_type}")
                print("-" * 60)
                
                # 尝试读取响应内容
                if 'application/json' in content_type:
                    # JSON 响应
                    try:
                        data = await response.json()
                        print("响应内容 (JSON):")
                        print(json.dumps(data, indent=2, ensure_ascii=False))
                    except Exception as e:
                        print(f"解析 JSON 失败: {e}")
                        text = await response.text()
                        print(f"原始响应文本: {text[:500]}")
                else:
                    # 文本或流式响应
                    print("响应内容 (文本/流式):")
                    buffer = ""
                    line_count = 0
                    async for chunk in response.content.iter_chunked(1024):
                        buffer += chunk.decode('utf-8', errors='ignore')
                        # 显示前 20 行
                        while '\n' in buffer and line_count < 20:
                            line, buffer = buffer.split('\n', 1)
                            line_count += 1
                            print(f"  [{line_count}] {line[:200]}")
                    
                    # 如果还有剩余内容
                    if buffer.strip() and line_count < 20:
                        print(f"  [{line_count + 1}] {buffer.strip()[:200]}")
                    
                    # 显示总长度
                    full_text = await response.text()
                    print(f"\n总响应长度: {len(full_text)} 字符")
                    if len(full_text) > 0:
                        print(f"前 500 字符: {full_text[:500]}")
                
                print("=" * 60)
                print("测试完成")
                print("=" * 60)
                
    except aiohttp.ClientError as e:
        print(f"请求错误: {e}")
        print("=" * 60)
    except asyncio.TimeoutError:
        print("请求超时（60秒）")
        print("=" * 60)
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)


# 如果直接运行此文件，执行测试
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python main.py <api_url> <api_key> [model]")
        print("示例: python main.py https://grsai.dakka.com.cn/v1/video/sora-video your_api_key sora-2")
        sys.exit(1)
    
    api_url = sys.argv[1]
    api_key = sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "sora-2"
    
    asyncio.run(test_sora_api(api_url, api_key, model))
