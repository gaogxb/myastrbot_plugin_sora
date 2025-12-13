"""
独立的 sora2 API 测试脚本
不依赖 astrbot 框架，可以直接运行
"""
import asyncio
import json
import aiohttp


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


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python test_standalone.py <api_url> <api_key> [model]")
        print("示例: python test_standalone.py https://grsai.dakka.com.cn/v1/video/sora-video your_api_key sora-2")
        print("\n或者直接修改脚本中的参数运行")
        # 使用默认参数
        asyncio.run(test_sora_api(
            api_url="https://grsai.dakka.com.cn/v1/video/sora-video",
            api_key="",
            model="sora-2"
        ))
    else:
        api_url = sys.argv[1]
        api_key = sys.argv[2]
        model = sys.argv[3] if len(sys.argv) > 3 else "sora-2"
        asyncio.run(test_sora_api(api_url, api_key, model))

