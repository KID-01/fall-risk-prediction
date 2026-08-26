"""
音频采集模块 - 支持麦克风 / RTSP / 本地文件三种音频源
产出固定时长的 AudioChunk (float32 单声道, 目标采样率), 供 AudioAnalyzer 直接消费

设计约束:
- 懒加载: sounddevice / ffmpeg 仅在首次使用时导入
- 线程安全: stop_event 在 <=200ms 子块内检查, 保证 stop() 响应性
- 统一契约: 所有后端产出 AudioChunk(waveform: float32 mono, sample_rate, timestamp, duration_sec)
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import BinaryIO

import numpy as np
import soundfile as sf

from src.utils.logger import get_logger

log = get_logger(__name__)

# 懒加载占位符, 供测试 patch
sd = None


# ============================================================
# 公共契约
# ============================================================
@dataclass(frozen=True)
class AudioChunk:
    """固定时长的音频块, 已归一化为 float32 单声道 + 目标采样率"""

    waveform: np.ndarray          # float32, 单声道 (N,)
    sample_rate: int              # 目标采样率 (如 32000)
    timestamp: float              # 块起始时间(秒, 相对监控启动)
    duration_sec: float           # 实际时长(秒)


# ============================================================
# 内部工具
# ============================================================
def _resample_to_target(wave: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    """重采样到目标采样率 (使用 librosa, 已在依赖中)"""
    if src_sr == target_sr:
        return wave
    try:
        import librosa
        return librosa.resample(wave, orig_sr=src_sr, target_sr=target_sr).astype(np.float32)
    except Exception as e:
        log.warning(f"librosa 重采样失败: {e}, 回退线性插值")
        # 简单线性插值回退
        ratio = target_sr / src_sr
        new_len = int(len(wave) * ratio)
        return np.interp(
            np.arange(new_len) / ratio,
            np.arange(len(wave)),
            wave,
        ).astype(np.float32)


def _to_float32_mono(wave: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    """归一化: float32 + 单声道 + 目标采样率"""
    # 1. 确保 float32
    if wave.dtype != np.float32:
        if wave.dtype == np.int16:
            wave = wave.astype(np.float32) / 32768.0
        elif wave.dtype == np.int32:
            wave = wave.astype(np.float32) / 2147483648.0
        else:
            wave = wave.astype(np.float32)

    # 2. 双声道 -> 单声道 (取均值)
    if wave.ndim == 2:
        wave = wave.mean(axis=1)

    # 3. 重采样
    wave = _resample_to_target(wave, src_sr, target_sr)

    return wave.astype(np.float32)


# ============================================================
# 音频采集器
# ============================================================
class AudioCapture:
    """
    音频采集器, 支持三种源:
    - "mic"           -> 系统麦克风
    - "rtsp://..."    -> RTSP 音频流 (通过 ffmpeg 解码)
    - "/path/file.wav" -> 本地音频文件
    - "auto"          -> 由 monitor.py 根据视频源推导

    产出: 固定时长 AudioChunk, 已归一化为 float32 单声道目标采样率
    """

    def __init__(
        self,
        source: str,
        sample_rate: int = 32000,
        chunk_seconds: int = 10,
        input_device: int | None = None,
        ffmpeg_path: str = "",
        stop_event: threading.Event | None = None,
    ):
        """
        Args:
            source: 音频源 ("mic" | "rtsp://..." | 文件路径 | "auto")
            sample_rate: 目标采样率 (默认 32000, PANNs 要求)
            chunk_seconds: 切片时长(秒)
            input_device: 麦克风设备编号 (None=默认)
            ffmpeg_path: ffmpeg 可执行文件路径 (空=从 PATH 查找)
            stop_event: 外部停止事件, 用于跨线程优雅退出
        """
        self.source = source
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.input_device = input_device
        self.ffmpeg_path = ffmpeg_path or which("ffmpeg") or "ffmpeg"
        self.stop_event = stop_event or threading.Event()

        # 内部状态
        self._file: BinaryIO | None = None
        self._file_sf: sf.SoundFile | None = None
        self._ffmpeg_proc: subprocess.Popen | None = None
        self._mic_stream = None
        self._is_open = False
        self._start_time: float | None = None
        self._total_read_sec = 0.0

    # ========================================================
    # 公共接口
    # ========================================================
    def open(self) -> bool:
        """打开音频源, 返回是否成功"""
        if self._is_open:
            return True

        self.stop_event.clear()
        src = self.source.lower()

        try:
            if src == "mic":
                return self._open_mic()
            elif src.startswith("rtsp://") or src.startswith("rtmp://"):
                return self._open_network_stream()
            elif src == "auto" or src == "off":
                log.info(f"音频源 '{src}' 暂不直接打开, 由 monitor.py 决定")
                self._is_open = True  # 标记为"已处理", 实际采集由 monitor 决定
                return True
            else:
                # 视为文件路径
                return self._open_file()
        except Exception as e:
            log.error(f"打开音频源失败: {e}")
            self.close()
            return False

    def read_chunk(self) -> AudioChunk | None:
        """阻塞读取一个 chunk_seconds 的音频块, 返回 None 表示流结束/出错/停止"""
        if not self._is_open:
            return None

        if self.stop_event.is_set():
            return None

        # 根据源类型分发
        if self.source.lower() == "mic":
            return self._read_mic_chunk()
        elif self.source.lower().startswith("rtsp://") or self.source.lower().startswith("rtmp://"):
            return self._read_rtsp_chunk()
        elif self.source.lower() == "auto" or self.source.lower() == "off":
            return None  # auto/off 由 monitor 处理
        else:
            return self._read_file_chunk()

    def chunks(self) -> Iterator[AudioChunk]:
        """迭代器: 持续产出 AudioChunk 直到流结束/出错/停止"""
        if not self._is_open:
            if not self.open():
                raise RuntimeError(f"无法打开音频源: {self.source}")

        self._start_time = time.time()
        self._total_read_sec = 0.0
        self._base_timestamp = time.time()

        try:
            while not self.stop_event.is_set():
                chunk = self.read_chunk()
                if chunk is None:
                    break
                yield chunk
        finally:
            # 迭代器结束/异常时不自动 close, 由外部控制
            pass

    def close(self):
        """释放资源, 幂等"""
        self.stop_event.set()

        if self._mic_stream is not None:
            try:
                self._mic_stream.stop()
                self._mic_stream.close()
            except Exception:
                pass
            self._mic_stream = None

        if self._ffmpeg_proc is not None:
            # 先关闭 stdout 解除阻塞读取, 再终止进程
            try:
                if self._ffmpeg_proc.stdout:
                    self._ffmpeg_proc.stdout.close()
            except Exception:
                pass
            try:
                self._ffmpeg_proc.terminate()
                self._ffmpeg_proc.wait(timeout=2)
            except Exception:
                try:
                    self._ffmpeg_proc.kill()
                    self._ffmpeg_proc.wait(timeout=1)
                except Exception:
                    pass
            self._ffmpeg_proc = None

        if self._file_sf is not None:
            try:
                self._file_sf.close()
            except Exception:
                pass
            self._file_sf = None

        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

        self._is_open = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ========================================================
    # 内部: 源打开
    # ========================================================
    def _open_mic(self) -> bool:
        """打开麦克风 (sounddevice)"""
        try:
            import sounddevice as sd
        except ImportError as e:
            log.error(f"sounddevice 未安装: {e}")
            return False

        try:
            device_info = sd.query_devices(self.input_device, "input") if self.input_device is not None else sd.query_devices(None, "input")
            log.info(f"使用麦克风设备: {device_info['name']} (索引 {device_info['index']})")
        except Exception as e:
            log.warning(f"查询麦克风设备失败: {e}")

        try:
            self._mic_stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.input_device,
                blocksize=0,  # 由 sounddevice 自动决定
            )
            self._mic_stream.start()
            self._is_open = True
            log.info("麦克风已启动")
            return True
        except Exception as e:
            log.error(f"启动麦克风失败: {e}")
            return False

    def _open_network_stream(self) -> bool:
        """打开 RTSP/RTMP 网络音频流 (通过 ffmpeg 解码为 PCM s16le stdout)"""
        if not which(self.ffmpeg_path):
            log.error(f"ffmpeg 不存在: {self.ffmpeg_path}")
            return False

        is_rtmp = self.source.lower().startswith("rtmp://")

        # RTSP 用 tcp 传输避免 UDP 丢包; RTMP 不需要此参数
        input_opts = ["-nostdin"]
        if not is_rtmp:
            input_opts.extend(["-rtsp_transport", "tcp"])

        cmd = [
            self.ffmpeg_path,
            *input_opts,
            "-i", self.source,
            "-vn",                      # 不要视频
            "-acodec", "pcm_s16le",     # 输出 16-bit PCM
            "-ar", str(self.sample_rate),  # 目标采样率
            "-ac", "1",                 # 单声道
            "-f", "s16le",              # 原始 PCM 格式
            "-loglevel", "warning",     # 输出警告+错误, 便于调试
            "-",                        # 输出到 stdout
        ]

        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW
        else:
            creationflags = 0

        try:
            self._ffmpeg_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=1024 * 1024,
                creationflags=creationflags,
            )
            time.sleep(1.0)
            if self._ffmpeg_proc.poll() is not None:
                log.error(
                    f"ffmpeg 进程异常退出 (code={self._ffmpeg_proc.returncode}): "
                    f"source={self.source[:80]}"
                )
                return False
            self._is_open = True
            protocol = "RTMP" if is_rtmp else "RTSP"
            log.info(f"{protocol} 音频流已连接: {self.source[:60]}...")
            return True
        except Exception as e:
            log.error(f"启动 ffmpeg 失败: {e}")
            return False

    def _open_file(self) -> bool:
        """打开本地音频文件"""
        path = Path(self.source)
        if not path.is_file():
            log.error(f"音频文件不存在: {path}")
            return False

        try:
            self._file_sf = sf.SoundFile(str(path), mode="r")
            src_sr = self._file_sf.samplerate
            log.info(f"打开音频文件: {path} (sr={src_sr}, ch={self._file_sf.channels}, frames={len(self._file_sf)})")
            self._is_open = True
            return True
        except Exception as e:
            log.error(f"打开音频文件失败: {e}")
            return False

    # ========================================================
    # 内部: 读取 chunk
    # ========================================================
    def _read_mic_chunk(self) -> AudioChunk | None:
        """从麦克风读取一个 chunk (内部按 <=200ms 子块累积, 检查 stop_event)"""
        if self._mic_stream is None:
            return None

        target_samples = self.sample_rate * self.chunk_seconds
        accumulated = []
        accumulated_samples = 0

        while accumulated_samples < target_samples and not self.stop_event.is_set():
            # 读取 <=200ms
            read_samples = min(self.sample_rate // 5, target_samples - accumulated_samples)  # 200ms
            try:
                data, overflowed = self._mic_stream.read(read_samples)
                if overflowed:
                    log.warning("麦克风缓冲区溢出")
                if len(data) == 0:
                    time.sleep(0.01)
                    continue
                accumulated.append(data)
                accumulated_samples += len(data)
            except KeyboardInterrupt:
                # stop_event 触发的中断
                break
            except Exception as e:
                log.error(f"麦克风读取失败: {e}")
                return None

        if self.stop_event.is_set() or accumulated_samples == 0:
            return None

        # 拼接并归一化
        wave = np.concatenate(accumulated)
        wave = wave.astype(np.float32)
        if wave.ndim == 2:
            wave = wave.mean(axis=1)

        timestamp = self._base_timestamp + self._total_read_sec
        self._total_read_sec += accumulated_samples / self.sample_rate
        duration = accumulated_samples / self.sample_rate

        return AudioChunk(
            waveform=wave,
            sample_rate=self.sample_rate,
            timestamp=timestamp,
            duration_sec=duration,
        )

    def _read_rtsp_chunk(self) -> AudioChunk | None:
        """从 ffmpeg stdout 读取一个 chunk (s16le -> float32)"""
        if self._ffmpeg_proc is None or self._ffmpeg_proc.stdout is None:
            return None

        target_samples = self.sample_rate * self.chunk_seconds
        bytes_needed = target_samples * 2  # s16le = 2 bytes/sample

        # 分片读取, 每次 <=200ms 检查 stop_event
        accumulated = bytearray()
        while len(accumulated) < bytes_needed and not self.stop_event.is_set():
            if self._ffmpeg_proc.poll() is not None:
                log.info("ffmpeg 进程已退出")
                return None

            # 每次读取 <=200ms 对应的字节
            read_bytes = min(self.sample_rate // 5 * 2, bytes_needed - len(accumulated))
            data = self._ffmpeg_proc.stdout.read(read_bytes)
            if not data:
                time.sleep(0.01)
                continue
            accumulated.extend(data)

        if self.stop_event.is_set() or len(accumulated) == 0:
            return None

        # s16le -> float32 mono
        wave = np.frombuffer(accumulated, dtype=np.int16).astype(np.float32) / 32768.0

        timestamp = self._base_timestamp + self._total_read_sec
        self._total_read_sec += len(wave) / self.sample_rate
        duration = len(wave) / self.sample_rate

        return AudioChunk(
            waveform=wave,
            sample_rate=self.sample_rate,
            timestamp=timestamp,
            duration_sec=duration,
        )

    def _read_file_chunk(self) -> AudioChunk | None:
        """从文件读取一个 chunk (按 chunk_seconds 切片, 自动重采样)"""
        if self._file_sf is None:
            return None

        src_sr = self._file_sf.samplerate
        target_samples = self.sample_rate * self.chunk_seconds
        # 源采样率下需要读取的样本数 (近似)
        src_samples_needed = int(target_samples * src_sr / self.sample_rate) + 1

        try:
            data = self._file_sf.read(src_samples_needed, dtype="float32", always_2d=True)
            if len(data) == 0:
                return None
        except Exception as e:
            log.error(f"读取音频文件失败: {e}")
            return None

        # 归一化: float32 mono + 目标采样率
        wave = _to_float32_mono(data, src_sr, self.sample_rate)

        timestamp = self._base_timestamp + self._total_read_sec
        self._total_read_sec += len(wave) / self.sample_rate
        duration = len(wave) / self.sample_rate

        return AudioChunk(
            waveform=wave,
            sample_rate=self.sample_rate,
            timestamp=timestamp,
            duration_sec=duration,
        )
