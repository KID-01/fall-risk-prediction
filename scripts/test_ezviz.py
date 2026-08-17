"""
T1.3 验证脚本 — 萤石开放平台 SDK 集成测试

运行方式:
    cd e:\fengxian\fall-risk-prediction
    venv\\Scripts\\python.exe scripts\test_ezviz.py

使用方法:
    1. 登录 https://open.ys7.com 并进入"控制台" → "我的应用"
    2. 将 appKey 和 appSecret 写入被 Git 忽略的 configs/ezviz.yaml
    3. 运行本脚本验证
"""
import asyncio
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ezviz import EzvizClient
from src.utils.config import get_ezviz_config


async def main():
    print("=" * 60)
    print("  T1.3 萤石开放平台 SDK 集成 — 验证测试")
    print("=" * 60)

    # 1. 加载配置
    print("\n[1] 加载配置文件...")
    try:
        cfg = get_ezviz_config()
        app_key = cfg.ezviz.app_key
        app_secret = cfg.ezviz.app_secret
    except Exception as e:
        print(f"  ❌ 配置加载失败: {e}")
        return

    if not app_key or app_key == "你的appKey填在这里":
        print("  ⚠️  请先在 configs/ezviz.yaml 中填入真实的 appKey 和 appSecret")
        print("     获取方式: 登录 https://open.ys7.com → 控制台 → 我的应用")
        return

    print(f"  ✅ 配置加载成功 (appKey: {app_key[:8]}...)")

    # 2. 创建客户端
    print("\n[2] 创建 EzvizClient...")
    client = EzvizClient(app_key=app_key, app_secret=app_secret)

    try:
        # 3. 获取 Token
        print("\n[3] 获取 AccessToken...")
        token = await client.get_token()
        print(f"  ✅ Token 获取成功: {token[:20]}...")
    except Exception as e:
        print(f"  ❌ Token 获取失败: {e}")
        return

    try:
        # 4. 获取设备列表
        print("\n[4] 获取设备列表...")
        devices = await client.list_devices()
        if devices:
            print(f"  ✅ 共 {len(devices)} 台设备:")
            for d in devices:
                status = "在线" if d.get("status") == 1 else "离线"
                print(f"     - {d.get('deviceName', '未知')} ({d.get('deviceSerial', '?')}) [{status}]")
        else:
            print("  ⚠️  当前账号下没有设备，请先激活设备套餐")
            print("     激活文档: https://ezsuperfans.com/portal.php?mod=view&aid=716")
            print("     激活码已存储在 configs/ezviz.yaml 中")
    except Exception as e:
        print(f"  ❌ 设备列表获取失败: {e}")

    try:
        # 5. 获取直播流（如果有设备）
        if devices:
            print("\n[5] 获取直播流地址...")
            first_device = devices[0]
            serial = first_device.get("deviceSerial", "")
            if serial:
                url = await client.get_hls_url(serial)
                if url:
                    print(f"  ✅ HLS 地址: {url[:60]}...")
                else:
                    print("  ⚠️  获取直播地址失败（可能设备离线或未开通云直播）")
    except Exception as e:
        print(f"  ❌ 直播流获取失败: {e}")

    finally:
        await client.close()

    print("\n" + "=" * 60)
    print("  验证完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
