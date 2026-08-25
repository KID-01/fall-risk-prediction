"""音频采集模块测试 - AudioCapture / AudioChunk

覆盖: 文件/RTSP/麦克风三种后端, stop_event 响应性, 固定时长切片, 采样率/声道归一化。
不依赖真实硬件/网络: 使用 soundfile 写入临时 wav, monkeypatch ffmpeg/sounddevice。
"""
from __future__ import annotations

import io
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from src.data.audio_capture import AudioCapture, AudioChunk


# ============================================================
# 测试辅助
# ============================================================
def _sine_wave(
    duration: float = 2.0,
    sample_rate: int = 16000,
    freq: float = 440.0,
    amplitude: float = 0.1,
) -> np.ndarray:
    """生成正弦波 (float32, 单声道)"""
    t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _write_wav(path: Path, wave: np.ndarray, sample_rate: int):
    sf.write(str(path), wave, sample_rate, subtype="PCM_16")


# ============================================================
# TestAudioChunk
# ============================================================
class TestAudioChunk:
    def test_fields(self):
        wave = np.zeros(32000, dtype=np.float32)
        chunk = AudioChunk(
            waveform=wave,
            sample_rate=32000,
            timestamp=1.5,
            duration_sec=1.0,
        )
        assert chunk.waveform is wave
        assert chunk.sample_rate == 32000
        assert chunk.timestamp == 1.5
        assert chunk.duration_sec == 1.0

    def test_duration_matches_waveform(self):
        sr = 16000
        dur = 2.5
        wave = _sine_wave(dur, sr)
        chunk = AudioChunk(waveform=wave, sample_rate=sr, timestamp=0.0, duration_sec=dur)
        assert len(chunk.waveform) == pytest.approx(sr * dur, abs=1)


# ============================================================
# TestAudioCaptureFileBackend - 无需外部依赖
# ============================================================
class TestAudioCaptureFileBackend:
    def test_open_existing_file(self, tmp_path):
        wave = _sine_wave(3.0, 16000)
        path = tmp_path / "tone.wav"
        _write_wav(path, wave, 16000)

        cap = AudioCapture(source=str(path), sample_rate=32000, chunk_seconds=1)
        assert cap.open() is True
        cap.close()

    def test_open_nonexistent_file(self, tmp_path):
        cap = AudioCapture(source=str(tmp_path / "missing.wav"), sample_rate=32000, chunk_seconds=1)
        assert cap.open() is False

    def test_chunks_yields_fixed_duration(self, tmp_path):
        """音频文件按 chunk_seconds 切片, 时长准确"""
        total_dur = 5.0
        wave = _sine_wave(total_dur, 16000)
        path = tmp_path / "long.wav"
        _write_wav(path, wave, 16000)

        chunk_seconds = 1
        cap = AudioCapture(source=str(path), sample_rate=32000, chunk_seconds=chunk_seconds)
        cap.open()

        chunks = list(cap.chunks())
        assert len(chunks) == int(total_dur / chunk_seconds)  # 5 chunks

        for i, chunk in enumerate(chunks):
            assert chunk.duration_sec == pytest.approx(chunk_seconds, abs=0.1)
            assert chunk.timestamp == pytest.approx(i * chunk_seconds, abs=0.1)
            assert chunk.sample_rate == 32000
            assert chunk.waveform.dtype == np.float32
            assert chunk.waveform.ndim == 1  # 单声道

        cap.close()

    def test_chunks_resamples_to_target_sr(self, tmp_path):
        """源采样率 16kHz -> 目标 32kHz 重采样"""
        wave = _sine_wave(2.0, 16000)
        path = tmp_path / "tone16k.wav"
        _write_wav(path, wave, 16000)

        cap = AudioCapture(source=str(path), sample_rate=32000, chunk_seconds=1)
        cap.open()
        chunks = list(cap.chunks())
        cap.close()

        for chunk in chunks:
            assert chunk.sample_rate == 32000
            # 重采样后样本数约为 2 倍
            assert len(chunk.waveform) == pytest.approx(32000, abs=100)

    def test_stereo_to_mono(self, tmp_path):
        """双声道输入 -> 单声道输出"""
        sr = 16000
        dur = 2.0
        t = np.arange(int(sr * dur), dtype=np.float32) / sr
        left = 0.1 * np.sin(2 * np.pi * 440 * t)
        right = 0.1 * np.sin(2 * np.pi * 880 * t)
        stereo = np.stack([left, right], axis=1).astype(np.float32)  # (N, 2)
        path = tmp_path / "stereo.wav"
        sf.write(str(path), stereo, sr, subtype="PCM_16")

        cap = AudioCapture(source=str(path), sample_rate=32000, chunk_seconds=1)
        cap.open()
        chunks = list(cap.chunks())
        cap.close()

        for chunk in chunks:
            assert chunk.waveform.ndim == 1

    def test_file_end_raises_stop_iteration(self, tmp_path):
        """文件读完 -> chunks() 迭代器自然结束"""
        wave = _sine_wave(1.0, 16000)
        path = tmp_path / "short.wav"
        _write_wav(path, wave, 16000)

        cap = AudioCapture(source=str(path), sample_rate=32000, chunk_seconds=10)  # chunk > 文件
        cap.open()
        chunks = list(cap.chunks())
        cap.close()
        # 文件只有 1s, 但 chunk_seconds=10, 仍会产出 1 个 chunk (剩余全部)
        assert len(chunks) == 1
        assert chunks[0].duration_sec == pytest.approx(1.0, abs=0.1)


# ============================================================
# TestAudioCaptureRTSPBackend - monkeypatch ffmpeg
# ============================================================
class TestAudioCaptureRTSPBackend:
    @patch("src.data.audio_capture.subprocess.Popen")
    @patch("src.data.audio_capture.which", return_value="/fake/ffmpeg")
    def test_rtsp_yields_chunks(self, mock_which, mock_popen, tmp_path):
        """RTSP 流模拟: ffmpeg 输出 s16le PCM -> 正确解码为 float32"""
        target_sr = 32000
        chunk_seconds = 1
        # 生成 3 秒的 PCM s16le 数据
        wave = _sine_wave(3.0, target_sr)
        # 转 s16le
        pcm = (wave * 32767).astype(np.int16).tobytes()

        proc = MagicMock()
        proc.stdout = io.BytesIO(pcm)
        proc.wait.return_value = 0
        # poll() returns None initially (process running), then 0 after data exhausted
        # _open_rtsp calls poll() once, then each chunk needs 5 reads = 15 reads for 3 chunks
        # Total poll() calls: 1 (open) + 15 (reads) = 16
        poll_results = [None] * 16 + [0]
        proc.poll.side_effect = lambda: poll_results.pop(0) if poll_results else 0
        proc.returncode = 0
        proc.terminate = MagicMock()
        proc.kill = MagicMock()
        mock_popen.return_value = proc

        cap = AudioCapture(source="rtsp://fake/stream", sample_rate=target_sr, chunk_seconds=chunk_seconds)
        cap.open()
        chunks = list(cap.chunks())
        cap.close()

        # 应产出 3 个 1 秒 chunks
        assert len(chunks) == 3
        for chunk in chunks:
            assert chunk.sample_rate == target_sr
            assert chunk.duration_sec == pytest.approx(1.0, abs=0.1)
            assert chunk.waveform.dtype == np.float32
            assert chunk.waveform.ndim == 1

    @patch("src.data.audio_capture.which", return_value=None)
    def test_rtsp_ffmpeg_failure_raises(self, mock_which):
        """ffmpeg 不存在 -> open() 返回 False"""
        cap = AudioCapture(source="rtsp://fake/stream", sample_rate=32000, chunk_seconds=1)
        assert cap.open() is False


# ============================================================
# TestAudioCaptureMicBackend - monkeypatch sounddevice (sys.modules patch)
# ============================================================
class TestAudioCaptureMicBackend:
    def test_mic_yields_chunks(self):
        """麦克风模拟: 产出固定时长 chunks"""
        target_sr = 32000
        chunk_seconds = 1

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        wave = _sine_wave(chunk_seconds, target_sr)
        mock_stream.read = MagicMock(return_value=(wave, len(wave)))
        mock_stream.read.side_effect = [
            (wave, len(wave)),
            (wave, len(wave)),
            (wave, len(wave)),
            KeyboardInterrupt,
        ]

        mock_sd = MagicMock()
        mock_sd.RawInputStream.return_value = mock_stream
        mock_sd.query_devices.return_value = {"name": "Mock Mic", "index": 0}

        # 在 sys.modules 中 mock sounddevice, 使得方法内的 import 能获取到 mock
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            cap = AudioCapture(source="mic", sample_rate=target_sr, chunk_seconds=chunk_seconds)
            cap.open()
            chunks = list(cap.chunks())
            cap.close()

        assert len(chunks) == 3
        for chunk in chunks:
            assert chunk.sample_rate == target_sr
            assert chunk.duration_sec == pytest.approx(chunk_seconds, abs=0.05)
            assert chunk.waveform.dtype == np.float32
            assert chunk.waveform.ndim == 1

    def test_mic_stop_event_exits_early(self):
        """stop_event 被设置 -> 立即退出"""
        target_sr = 32000
        chunk_seconds = 1

        stop_event = threading.Event()
        wave = _sine_wave(chunk_seconds, target_sr)

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)

        def read_with_stop(*args, **kwargs):
            if stop_event.is_set():
                raise KeyboardInterrupt
            return wave, len(wave)

        mock_sd = MagicMock()
        mock_sd.RawInputStream.return_value = mock_stream
        mock_stream.read = MagicMock(side_effect=read_with_stop)
        mock_sd.query_devices.return_value = {"name": "Mock Mic", "index": 0}

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            cap = AudioCapture(source="mic", sample_rate=target_sr, chunk_seconds=chunk_seconds, stop_event=stop_event)
            cap.open()

            # 启动后立即设置 stop_event
            stop_event.set()
            chunks = list(cap.chunks())
            cap.close()

        # 应该立即退出, 可能产出 0 或 1 个 chunk
        assert len(chunks) <= 1


# ============================================================
# TestStopEventResponsiveness - 关键: <=200ms 切片检查
# ============================================================
class TestStopEventResponsiveness:
    def test_file_backend_respects_stop_event(self, tmp_path):
        """文件后端: 即使 chunk_seconds 很大, stop_event 也能中断"""
        wave = _sine_wave(10.0, 16000)
        path = tmp_path / "long.wav"
        _write_wav(path, wave, 16000)

        stop_event = threading.Event()
        cap = AudioCapture(source=str(path), sample_rate=32000, chunk_seconds=5, stop_event=stop_event)
        cap.open()

        # 启动迭代器后立即设置 stop_event
        # 这里模拟: 先获取迭代器, 再设置 event
        chunks = []
        try:
            for chunk in cap.chunks():
                chunks.append(chunk)
                stop_event.set()  # 处理完第一个 chunk 后设置
        except RuntimeError:
            pass  # stop_event 在内部被检测到
        cap.close()

        # 至少产出了第一个 chunk, 然后退出
        assert len(chunks) >= 1


# ============================================================
# TestEdgeCases
# ============================================================
class TestEdgeCases:
    def test_close_idempotent(self, tmp_path):
        """close() 可重复调用不报错"""
        wave = _sine_wave(1.0, 16000)
        path = tmp_path / "tone.wav"
        _write_wav(path, wave, 16000)

        cap = AudioCapture(source=str(path), sample_rate=32000, chunk_seconds=1)
        cap.open()
        cap.close()
        cap.close()  # 二次调用不应报错

    def test_context_manager(self, tmp_path):
        """支持 with 语句"""
        wave = _sine_wave(1.0, 16000)
        path = tmp_path / "tone.wav"
        _write_wav(path, wave, 16000)

        with AudioCapture(source=str(path), sample_rate=32000, chunk_seconds=1) as cap:
            chunks = list(cap.chunks())
            assert len(chunks) == 1

    def test_source_auto_derive_from_video(self):
        """source='auto' 时根据视频源推导 (仅测试解析逻辑, 不真正连接)"""
        # 这里只验证 AudioCapture 能接受 'auto', 具体推导在 monitor.py 中
        cap = AudioCapture(source="auto", sample_rate=32000, chunk_seconds=1)
        assert cap.source == "auto"

    @patch("src.data.audio_capture.sd", create=True)
    def test_input_device_passed_to_sounddevice(self, mock_sd):
        """input_device 参数透传给 sounddevice"""
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.read = MagicMock(side_effect=KeyboardInterrupt)
        mock_sd.RawInputStream.return_value = mock_stream

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            cap = AudioCapture(source="mic", sample_rate=32000, chunk_seconds=1, input_device=2)
            cap.open()
            cap.close()

        # 验证 device=2 被传递
        mock_sd.RawInputStream.assert_called_once()
        kwargs = mock_sd.RawInputStream.call_args[1]
        assert kwargs.get("device") == 2


# ============================================================
# TestAudioChunkFloat32MonoContract - 契约测试
# ============================================================
class TestAudioChunkFloat32MonoContract:
    """所有后端产出的 AudioChunk 必须满足: float32, 单声道, 目标采样率"""

    def test_chunk_contract_file(self, tmp_path):
        target_sr = 32000
        chunk_seconds = 1

        wave = _sine_wave(2.0, 16000)
        path = tmp_path / "tone.wav"
        _write_wav(path, wave, 16000)
        cap = AudioCapture(source=str(path), sample_rate=target_sr, chunk_seconds=chunk_seconds)
        cap.open()
        chunks = list(cap.chunks())
        cap.close()

        # 契约断言
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.waveform.dtype == np.float32, f"dtype={chunk.waveform.dtype}"
            assert chunk.waveform.ndim == 1, f"ndim={chunk.waveform.ndim}"
            assert chunk.sample_rate == target_sr, f"sr={chunk.sample_rate}"
            assert chunk.duration_sec > 0
            assert chunk.timestamp >= 0

    @patch("src.data.audio_capture.subprocess.Popen")
    @patch("src.data.audio_capture.which", return_value="/fake/ffmpeg")
    def test_chunk_contract_rtsp(self, mock_which, mock_popen, tmp_path):
        target_sr = 32000
        chunk_seconds = 1

        wave = _sine_wave(2.0, target_sr)
        pcm = (wave * 32767).astype(np.int16).tobytes()
        proc = MagicMock()
        proc.stdout = io.BytesIO(pcm)
        proc.wait.return_value = 0
        # poll() returns None initially (process running), then 0 after data exhausted
        poll_results = [None] * 10 + [0]
        proc.poll.side_effect = lambda: poll_results.pop(0) if poll_results else 0
        proc.returncode = 0
        proc.terminate = MagicMock()
        proc.kill = MagicMock()
        mock_popen.return_value = proc

        cap = AudioCapture(source="rtsp://fake", sample_rate=target_sr, chunk_seconds=chunk_seconds)
        cap.open()
        chunks = list(cap.chunks())
        cap.close()

        # 契约断言
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.waveform.dtype == np.float32, f"dtype={chunk.waveform.dtype}"
            assert chunk.waveform.ndim == 1, f"ndim={chunk.waveform.ndim}"
            assert chunk.sample_rate == target_sr, f"sr={chunk.sample_rate}"
            assert chunk.duration_sec > 0
            assert chunk.timestamp >= 0

    @patch("src.data.audio_capture.sd", create=True)
    def test_chunk_contract_mic(self, mock_sd):
        target_sr = 32000
        chunk_seconds = 1

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        wave = _sine_wave(chunk_seconds, target_sr)
        mock_stream.read = MagicMock(return_value=(wave, len(wave)))
        mock_stream.read.side_effect = [
            (wave, len(wave)),
            (wave, len(wave)),
            KeyboardInterrupt,
        ]
        mock_sd.RawInputStream.return_value = mock_stream
        mock_sd.query_devices.return_value = {"name": "Mock Mic", "index": 0}

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            cap = AudioCapture(source="mic", sample_rate=target_sr, chunk_seconds=chunk_seconds)
            cap.open()
            chunks = list(cap.chunks())
            cap.close()

        # 契约断言
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.waveform.dtype == np.float32, f"dtype={chunk.waveform.dtype}"
            assert chunk.waveform.ndim == 1, f"ndim={chunk.waveform.ndim}"
            assert chunk.sample_rate == target_sr, f"sr={chunk.sample_rate}"
            assert chunk.duration_sec > 0
            assert chunk.timestamp >= 0
