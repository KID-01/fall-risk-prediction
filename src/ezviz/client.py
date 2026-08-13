"""
萤石开放平台 API 客户端

封装与萤石云的所有 HTTP 通信，包括：
- AccessToken 自动获取、缓存、过期刷新
- 设备列表查询
- 视频流地址获取（支持 HLS/RTSP）
- 云台控制（PTZ）
- 失败自动重试
"""
from __future__ import annotations

import time
import asyncio
from pathlib import Path
from typing import Any

import httpx
from loguru import logger
from omegaconf import OmegaConf


# ============================================================
# 萤石开放平台 API 端点
# ============================================================
BASE_URL = "https://open.ys7.com"
API_TOKEN = f"{BASE_URL}/api/lapp/token/get"
API_DEVICE_LIST = f"{BASE_URL}/api/lapp/device/list"
API_DEVICE_INFO = f"{BASE_URL}/api/lapp/device/info"
API_LIVE_ADDRESS = f"{BASE_URL}/api/lapp/v2/live/address/get"
API_PTZ_START = f"{BASE_URL}/api/lapp/device/ptz/start"
API_PTZ_STOP = f"{BASE_URL}/api/lapp/device/ptz/stop"

# Token 提前刷新时间（秒），避免在请求过程中过期
TOKEN_REFRESH_AHEAD = 300  # 5分钟


class EzvizClient:
    """
    萤石开放平台 API 客户端

    使用方式:
        client = EzvizClient(app_key="xxx", app_secret="xxx")
        await client.ensure_token()          # 获取 token
        devices = await client.list_devices() # 获取设备列表
        url = await client.get_live_stream("设备序列号")  # 获取直播地址
        await client.ptz_control("设备序列号", "up")      # 云台控制
    """

    def __init__(
        self,
        app_key: str,
        app_secret: str,
        base_url: str = BASE_URL,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Args:
            app_key: 萤石开放平台应用的 AppKey
            app_secret: 萤石开放平台应用的 AppSecret
            base_url: API 基础地址（默认国内站）
            max_retries: 失败重试次数
            retry_delay: 重试间隔（秒），使用指数退避
        """
        self.app_key = app_key
        self.app_secret = app_secret
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # 内部状态
        self._token: str | None = None
        self._token_expire_time: float = 0.0  # 过期时间戳
        self._client: httpx.AsyncClient | None = None

    # ── HTTP 客户端管理 ─────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        """获取或创建 HTTP 客户端（懒加载）"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),  # 30秒超时
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        return self._client

    async def close(self):
        """关闭 HTTP 客户端，释放连接"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ── Token 管理 ─────────────────────────────────────

    async def get_token(self) -> str:
        """
        获取有效的 AccessToken

        自动处理缓存：如果 token 还未过期，直接返回缓存的 token；
        如果已过期或即将过期，自动请求新 token。

        萤石 AccessToken 有效期通常为 7 天，这里提前 5 分钟刷新。

        Returns:
            accessToken 字符串
        """
        now = time.time()
        # 如果 token 还有效（提前 5 分钟刷新），直接返回
        if self._token is not None and now < self._token_expire_time - TOKEN_REFRESH_AHEAD:
            return self._token

        # 请求新 token
        logger.info("正在获取萤石 AccessToken...")
        client = await self._get_client()

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = await client.post(
                    API_TOKEN,
                    data={
                        "appKey": self.app_key,
                        "appSecret": self.app_secret,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

                if data.get("code") != "200":
                    error_msg = data.get("msg", "未知错误")
                    logger.error(f"获取 Token 失败: {error_msg}")
                    raise RuntimeError(f"萤石 API 返回错误: {error_msg}")

                self._token = data["data"]["accessToken"]
                expire_seconds = int(data["data"].get("expireTime", 604800))
                self._token_expire_time = now + expire_seconds

                logger.info(f"Token 获取成功，有效期 {expire_seconds // 86400} 天")
                return self._token

            except httpx.HTTPError as e:
                logger.warning(f"Token 请求失败 (第 {attempt}/{self.max_retries} 次): {e}")
                if attempt < self.max_retries:
                    wait = self.retry_delay * (2 ** (attempt - 1))  # 指数退避
                    await asyncio.sleep(wait)
                else:
                    raise RuntimeError(f"Token 请求失败，已重试 {self.max_retries} 次") from e

        # 理论上不会走到这里
        raise RuntimeError("Token 获取失败")

    async def ensure_token(self) -> str:
        """确保 token 可用（同 get_token，语义更明确的别名）"""
        return await self.get_token()

    # ── 设备管理 ───────────────────────────────────────

    async def list_devices(self, page_start: int = 0, page_size: int = 10) -> list[dict[str, Any]]:
        """
        获取账号下的设备列表

        Args:
            page_start: 分页起始位置（从 0 开始）
            page_size: 每页数量（最大 50）

        Returns:
            设备信息列表，每个设备包含:
            - deviceSerial: 设备序列号
            - deviceName: 设备名称
            - deviceType: 设备型号
            - status: 在线状态 (1=在线, 0=离线)
            - channelNo: 通道号
            - isEncrypt: 是否加密
        """
        token = await self.get_token()
        client = await self._get_client()

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = await client.post(
                    API_DEVICE_LIST,
                    data={
                        "accessToken": token,
                        "pageStart": page_start,
                        "pageSize": page_size,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

                if data.get("code") != "200":
                    logger.error(f"获取设备列表失败: {data.get('msg')}")
                    return []

                devices = data.get("data", [])
                logger.info(f"获取到 {len(devices)} 台设备")
                return devices

            except httpx.HTTPError as e:
                logger.warning(f"设备列表请求失败 (第 {attempt}/{self.max_retries} 次): {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** (attempt - 1)))
                else:
                    logger.error("设备列表请求最终失败")
                    return []

        return []

    async def get_device_info(self, device_serial: str) -> dict[str, Any] | None:
        """
        获取单个设备的详细信息

        Args:
            device_serial: 设备序列号

        Returns:
            设备详细信息字典，失败返回 None
        """
        token = await self.get_token()
        client = await self._get_client()

        try:
            resp = await client.post(
                API_DEVICE_INFO,
                data={"accessToken": token, "deviceSerial": device_serial},
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") == "200":
                return data.get("data")
            logger.error(f"获取设备信息失败: {data.get('msg')}")
            return None
        except httpx.HTTPError as e:
            logger.error(f"设备信息请求失败: {e}")
            return None

    # ── 视频流 ─────────────────────────────────────────

    async def get_live_stream(
        self,
        device_serial: str,
        channel_no: int = 1,
        protocol: int = 1,
    ) -> str | None:
        """
        获取设备直播流地址

        Args:
            device_serial: 设备序列号
            channel_no: 通道号（默认 1）
            protocol: 协议类型
                      1 = HLS（m3u8，适合 Web/H5 播放）
                      2 = RTMP
                      3 = FLV
                      4 = RTSP

        Returns:
            直播流 URL 字符串，失败返回 None
        """
        token = await self.get_token()
        client = await self._get_client()

        protocol_names = {1: "HLS", 2: "RTMP", 3: "FLV", 4: "RTSP"}
        proto_name = protocol_names.get(protocol, "Unknown")

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = await client.post(
                    API_LIVE_ADDRESS,
                    data={
                        "accessToken": token,
                        "deviceSerial": device_serial,
                        "channelNo": channel_no,
                        "protocol": protocol,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

                if data.get("code") != "200":
                    logger.error(f"获取直播地址失败: {data.get('msg')}")
                    return None

                # v2 API: data 是 dict，包含 url 字段
                raw = data["data"]
                if isinstance(raw, dict):
                    url = raw.get("url", "")
                elif isinstance(raw, list) and raw:
                    url = raw[0].get("url", "") if isinstance(raw[0], dict) else str(raw[0])
                else:
                    url = str(raw) if raw else ""
                logger.info(f"获取到 {proto_name} 直播地址: {url[:50]}...")
                return url

            except httpx.HTTPError as e:
                logger.warning(f"直播地址请求失败 (第 {attempt}/{self.max_retries} 次): {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** (attempt - 1)))
                else:
                    logger.error("直播地址请求最终失败")
                    return None

        return None

    async def get_hls_url(self, device_serial: str, channel_no: int = 1) -> str | None:
        """获取 HLS 流地址（适合 Web 播放）"""
        return await self.get_live_stream(device_serial, channel_no, protocol=1)

    async def get_rtsp_url(self, device_serial: str, channel_no: int = 1) -> str | None:
        """获取 RTSP 流地址（适合 OpenCV 拉流）"""
        return await self.get_live_stream(device_serial, channel_no, protocol=4)

    # ── 云台控制 (PTZ) ─────────────────────────────────

    # 云台控制方向常量
    DIRECTION_UP = 0        # 上
    DIRECTION_DOWN = 1      # 下
    DIRECTION_LEFT = 2      # 左
    DIRECTION_RIGHT = 3     # 右
    DIRECTION_UP_LEFT = 4   # 左上
    DIRECTION_UP_RIGHT = 5  # 右上
    DIRECTION_DOWN_LEFT = 6 # 左下
    DIRECTION_DOWN_RIGHT = 7# 右下

    async def ptz_start(
        self,
        device_serial: str,
        direction: int,
        speed: int = 1,
        channel_no: int = 1,
    ) -> bool:
        """
        开始云台控制（持续转动）

        Args:
            device_serial: 设备序列号
            direction: 方向 (0=上, 1=下, 2=左, 3=右)
            speed: 速度 (1=慢, 2=中, 3=快)
            channel_no: 通道号

        Returns:
            是否成功
        """
        token = await self.get_token()
        client = await self._get_client()

        try:
            resp = await client.post(
                API_PTZ_START,
                data={
                    "accessToken": token,
                    "deviceSerial": device_serial,
                    "channelNo": channel_no,
                    "direction": direction,
                    "speed": speed,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") == "200":
                logger.info(f"云台开始转动: 方向={direction}, 速度={speed}")
                return True
            logger.error(f"云台控制失败: {data.get('msg')}")
            return False
        except httpx.HTTPError as e:
            logger.error(f"云台控制请求失败: {e}")
            return False

    async def ptz_stop(
        self,
        device_serial: str,
        direction: int,
        channel_no: int = 1,
    ) -> bool:
        """
        停止云台控制

        Args:
            device_serial: 设备序列号
            direction: 需要停止的方向
            channel_no: 通道号

        Returns:
            是否成功
        """
        token = await self.get_token()
        client = await self._get_client()

        try:
            resp = await client.post(
                API_PTZ_STOP,
                data={
                    "accessToken": token,
                    "deviceSerial": device_serial,
                    "channelNo": channel_no,
                    "direction": direction,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") == "200":
                logger.info("云台停止")
                return True
            return False
        except httpx.HTTPError as e:
            logger.error(f"云台停止请求失败: {e}")
            return False

    async def ptz_control(
        self,
        device_serial: str,
        direction: str,
        duration_ms: int = 500,
        speed: int = 1,
    ) -> bool:
        """
        便捷云台控制：自动开始 → 等待 → 停止

        这是 T1.3 任务中要求封装的便捷方法。

        Args:
            device_serial: 设备序列号
            direction: 方向字符串 "up"/"down"/"left"/"right"
            duration_ms: 转动持续时间（毫秒）
            speed: 速度 (1-3)

        Returns:
            是否成功
        """
        direction_map = {
            "up": self.DIRECTION_UP,
            "down": self.DIRECTION_DOWN,
            "left": self.DIRECTION_LEFT,
            "right": self.DIRECTION_RIGHT,
        }

        direction_code = direction_map.get(direction.lower())
        if direction_code is None:
            logger.error(f"无效的方向: {direction}，支持: {list(direction_map.keys())}")
            return False

        # 开始转动
        ok = await self.ptz_start(device_serial, direction_code, speed)
        if not ok:
            return False

        # 等待指定时间
        await asyncio.sleep(duration_ms / 1000.0)

        # 停止转动
        await self.ptz_stop(device_serial, direction_code)
        return True

    # ── 上下文管理器 ───────────────────────────────────

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, *args):
        """异步上下文管理器出口"""
        await self.close()


# ============================================================
# 便捷工厂函数：从配置文件创建客户端
# ============================================================

def create_client_from_config(
    config_path: str | Path | None = None,
) -> EzvizClient:
    """
    从 YAML 配置文件创建 EzvizClient 实例

    按以下优先级查找配置:
    1. 传入的 config_path
    2. 项目 configs/ezviz.yaml

    Args:
        config_path: ezviz.yaml 文件路径

    Returns:
        配置好的 EzvizClient 实例
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "ezviz.yaml"

    if isinstance(config_path, str):
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"萤石配置文件不存在: {config_path}\n"
            f"请先创建该文件，参考 configs/ezviz.yaml 模板"
        )

    cfg = OmegaConf.load(config_path)
    ezviz_cfg = cfg.ezviz

    app_key = ezviz_cfg.app_key
    app_secret = ezviz_cfg.app_secret

    if not app_key or app_key == "你的appKey填在这里":
        raise ValueError(
            f"请先在 {config_path} 中填入真实的 appKey 和 appSecret\n"
            f"获取方式：登录 https://open.ys7.com → 控制台 → 我的应用"
        )

    return EzvizClient(app_key=app_key, app_secret=app_secret)