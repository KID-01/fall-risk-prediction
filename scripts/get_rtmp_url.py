"""获取萤石摄像头带认证的 RTMP 直播地址"""
import asyncio
import sys
sys.path.insert(0, ".")

from src.ezviz.client import EzvizClient
from src.utils.config import get_ezviz_config


async def main():
    config = get_ezviz_config()
    app_key = config.ezviz.app_key
    app_secret = config.ezviz.app_secret

    if not app_key or not app_secret:
        print("❌ 请在 configs/ezviz.yaml 中配置 app_key 和 app_secret")
        return

    client = EzvizClient(app_key=app_key, app_secret=app_secret)

    # 获取设备列表
    devices = await client.list_devices()
    if not devices:
        print("❌ 未找到设备，请检查 configs/ezviz.yaml 配置")
        return

    print(f"找到 {len(devices)} 个设备:\n")
    for d in devices:
        print(f"  设备名: {d.get('deviceName', '未知')}")
        print(f"  序列号: {d.get('deviceSerial', '未知')}")
        print(f"  状态: {'在线' if d.get('status') == 1 else '离线'}")
        print()

    # 使用第一个设备获取 RTMP 地址
    device_serial = devices[0].get("deviceSerial")
    if not device_serial:
        return

    print(f"正在获取设备 {device_serial} 的直播地址...\n")

    for proto, name in [(2, "RTMP"), (4, "RTSP"), (3, "FLV")]:
        url = await client.get_live_stream(device_serial, protocol=proto)
        if url:
            print(f"[{name}] {url}\n")
        else:
            print(f"[{name}] 获取失败\n")


if __name__ == "__main__":
    asyncio.run(main())