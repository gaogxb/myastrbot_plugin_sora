import asyncio
from main import test_sora_api

asyncio.run(test_sora_api(
    api_url="https://grsai.dakka.com.cn/v1/video/sora-video",
    api_key="sk-026657a2098f484194ffd9df7c8dc9b3",
    model="sora-2"
))