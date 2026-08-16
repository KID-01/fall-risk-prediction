"""将萤石摄像头视频编码切换为 H.264"""
import asyncio
import sys
sys.path.insert(0, ".")

from src.ezviz.client import EzvizClient
from src.utils.config import get_ezviz_config


async def main():
    config = get_ezviz_config()
    app_key = config.ezviz.app_key
    app_secret = config.ezviz.app_secret

    client = EzvizClient(app_key=app_key, app_secret=app_secret)

    devices = await client.list_devices()
    if not devices:
        print("❌ 未找到设备")
        return

    device_serial = devices[0].get("deviceSerial")
    device_name = devices[0].get("deviceName", "未知")
    print(f"设备: {device_name} ({device_serial})")

    ok = await client.set_video_encode(device_serial, encode_type="H264", stream_type=1)
    if ok:
        print("✅ 视频编码已切换为 H.264 (主码流)")
        print("   等待约 10 秒后，重新在前端启动监控即可")
    else:
        print("❌ 切换失败，请检查设备是否在线或是否支持编码切换")


if __name__ == "__main__":
    asyncio.run(main())