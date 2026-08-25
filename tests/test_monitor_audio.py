"""monitor.py 音频集成测试"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

from src.alerts.engine import RiskLevel
from src.inference.audio_analyzer import AudioAnalysisResult, AudioEvent, SoundCategory
from src.inference.monitor import FallRiskMonitor, MonitorStatus

# ── MonitorStatus 音频字段 ──


class TestMonitorStatusAudioFields:
    """MonitorStatus 应包含所有音频字段且默认值正确"""

    def test_default_values(self):
        s = MonitorStatus()
        assert s.audio_enabled is False
        assert s.audio_source == ""
        assert s.last_audio_result is None
        assert s.audio_chunks_processed == 0
        assert s.audio_error is None
        assert s._pending_audio_events == []
        assert isinstance(s._audio_lock, type(threading.Lock()))

    def test_set_values(self):
        event = AudioEvent(
            category=SoundCategory.IMPACT, label="fall", class_index=1, score=0.95, timestamp=1.0
        )
        result = AudioAnalysisResult(
            events=[event], top_labels=[("fall", 0.95)], duration_sec=10.0, elapsed_ms=120.0
        )
        s = MonitorStatus(
            audio_enabled=True,
            audio_source="mic",
            last_audio_result=result,
            audio_chunks_processed=3,
        )
        assert s.audio_enabled is True
        assert s.audio_source == "mic"
        assert s.last_audio_result is result
        assert s.audio_chunks_processed == 3


# ── start/stop 音频线程 ──


class TestMonitorStartStopAudio:
    """start() 应按 audio_source 启动/跳过音频线程"""

    @patch.object(FallRiskMonitor, "__new__", lambda cls: object.__new__(cls))
    def _make(self):
        m = FallRiskMonitor.__new__(FallRiskMonitor)
        m._initialized = False
        m.config = MagicMock()
        m.config.get.return_value = {"source": "off", "enabled": True}
        m.config.audio.sample_rate = 32000
        m.config.audio.chunk_seconds = 10
        m.config.audio.get.return_value = ""
        m.config.pose_estimation = {"backend": "mediapipe"}
        m.status = MonitorStatus()
        m._stop_flag = threading.Event()
        m._keypoint_buffer = []
        m._buffer_window = 30
        m._thread = None
        m._audio_thread = None
        m.audio_analyzer = None
        m.audio_capture = None
        m.video_capture = None
        m.human_detector = MagicMock()
        m.keypoint_extractor = MagicMock()
        m.frame_filter = MagicMock()
        m.feature_calculator = MagicMock()
        m.baseline_manager = MagicMock()
        m.deviation_detector = MagicMock()
        m.alert_engine = MagicMock()
        return m

    @patch("src.inference.monitor.AudioCapture")
    @patch("src.inference.monitor.AudioAnalyzer")
    @patch("src.inference.monitor.threading.Thread")
    def test_start_with_mic_spawns_audio_thread(self, mock_thread, mock_analyzer, mock_capture):
        m = self._make()
        mock_thread.return_value = MagicMock()
        result = m.start(source="0", audio_source="mic")
        assert result is True
        assert m.status.audio_enabled is True
        assert m.status.audio_source == "mic"
        mock_analyzer.assert_called_once()
        mock_capture.assert_called_once()

    def test_start_without_audio_skips(self):
        m = self._make()
        result = m.start(source="0")
        assert result is True
        assert m.status.audio_enabled is False
        assert m.audio_analyzer is None
        assert m._audio_thread is None

    def test_start_with_config_off_skips(self):
        m = self._make()
        m.config.get.return_value = {"source": "off", "enabled": True}
        m.start(source="0")
        assert m.status.audio_enabled is False

    @patch("src.inference.monitor.AudioCapture")
    @patch("src.inference.monitor.AudioAnalyzer")
    @patch("src.inference.monitor.threading.Thread")
    def test_stop_joins_audio_thread(self, mock_thread, mock_analyzer, mock_capture):
        m = self._make()
        mock_audio_thread = MagicMock()
        m._audio_thread = mock_audio_thread
        mock_capture = MagicMock()
        m.audio_capture = mock_capture
        m.stop()
        mock_audio_thread.join.assert_called_once_with(timeout=5)
        assert m._audio_thread is None
        mock_capture.close.assert_called_once()
        assert m.audio_capture is None


# ── _run_audio 循环 ──


class TestRunAudio:
    """_run_audio 应遍历 chunks、调用 analyzer、存入 pending_events"""

    def test_processes_chunks_and_stores_events(self):
        m = object.__new__(FallRiskMonitor)
        m._stop_flag = threading.Event()
        m.audio_analyzer = MagicMock()
        m.status = MonitorStatus()

        event = AudioEvent(
            category=SoundCategory.IMPACT, label="fall", class_index=1, score=0.9, timestamp=1.0
        )
        result = AudioAnalysisResult(events=[event], top_labels=[], duration_sec=10.0, elapsed_ms=50.0)
        m.audio_analyzer.analyze_waveform.return_value = result

        mock_chunk = MagicMock()
        mock_chunk.waveform = [0.0]
        mock_chunk.sample_rate = 32000
        mock_chunk.timestamp = 1.0

        mock_capture = MagicMock()
        mock_capture.__enter__ = MagicMock(return_value=mock_capture)
        mock_capture.__exit__ = MagicMock(return_value=False)
        mock_capture.chunks.return_value = [mock_chunk]
        m.audio_capture = mock_capture

        m._run_audio()

        m.audio_analyzer.analyze_waveform.assert_called_once()
        assert m.status.audio_chunks_processed == 1
        assert len(m.status._pending_audio_events) == 1
        assert m.status.last_audio_result is result

    def test_analyzer_exception_stored(self):
        m = object.__new__(FallRiskMonitor)
        m._stop_flag = threading.Event()
        m.status = MonitorStatus()

        mock_chunk = MagicMock()
        mock_chunk.waveform = [0.0]
        mock_chunk.sample_rate = 32000
        mock_chunk.timestamp = 1.0

        mock_capture = MagicMock()
        mock_capture.__enter__ = MagicMock(return_value=mock_capture)
        mock_capture.__exit__ = MagicMock(return_value=False)
        mock_capture.chunks.return_value = [mock_chunk]
        m.audio_capture = mock_capture

        m.audio_analyzer = MagicMock()
        m.audio_analyzer.analyze_waveform.side_effect = RuntimeError("model load fail")

        m._run_audio()

        assert m.status.audio_error == "model load fail"
        assert m.status.audio_chunks_processed == 0

    def test_stop_flag_breaks_loop(self):
        m = object.__new__(FallRiskMonitor)
        m.status = MonitorStatus()
        m.audio_analyzer = MagicMock()

        stop = threading.Event()
        stop.set()
        m._stop_flag = stop

        call_count = 0

        def slow_chunks():
            nonlocal call_count
            while not stop.is_set():
                call_count += 1
                yield MagicMock(waveform=[0.0], sample_rate=32000, timestamp=0.0)

        mock_capture = MagicMock()
        mock_capture.__enter__ = MagicMock(return_value=mock_capture)
        mock_capture.__exit__ = MagicMock(return_value=False)
        mock_capture.chunks.return_value = slow_chunks()
        m.audio_capture = mock_capture

        m._run_audio()
        m.audio_analyzer.analyze_waveform.assert_not_called()


# ── _run 阶段7 排空音频事件 ──


class TestRunStage7AudioDrain:
    """_run 阶段7 应排空 _pending_audio_events 传入 AlertEngine"""

    def test_drains_pending_events(self):
        m = object.__new__(FallRiskMonitor)
        m.status = MonitorStatus()
        event = AudioEvent(
            category=SoundCategory.IMPACT, label="fall", class_index=1, score=0.9, timestamp=1.0
        )
        m.status._pending_audio_events.append(event)
        m.alert_engine = MagicMock()
        m.alert_engine.evaluate.return_value = MagicMock(level=RiskLevel.LOW, message="ok")

        with m.status._audio_lock:
            pending = list(m.status._pending_audio_events)
            m.status._pending_audio_events.clear()

        m.alert_engine.evaluate(
            MagicMock(), 1.0, has_activity=True, audio_events=pending or None
        )

        assert len(pending) == 1
        assert m.status._pending_audio_events == []
        m.alert_engine.evaluate.assert_called_once()

    def test_empty_events_passes_none(self):
        m = object.__new__(FallRiskMonitor)
        m.status = MonitorStatus()
        m.alert_engine = MagicMock()
        m.alert_engine.evaluate.return_value = MagicMock(level=RiskLevel.LOW, message="ok")

        with m.status._audio_lock:
            pending = list(m.status._pending_audio_events)
            m.status._pending_audio_events.clear()

        m.alert_engine.evaluate(
            MagicMock(), 1.0, has_activity=True, audio_events=pending or None
        )

        call_kwargs = m.alert_engine.evaluate.call_args
        assert call_kwargs[1]["audio_events"] is None


# ── get_status 音频字段 ──


class TestGetStatusAudio:
    """get_status() 应返回所有音频字段"""

    def test_returns_audio_fields(self):
        m = object.__new__(FallRiskMonitor)
        m.status = MonitorStatus(
            audio_enabled=True,
            audio_source="mic",
            audio_chunks_processed=5,
            audio_error="some error",
            last_audio_result=AudioAnalysisResult(
                events=[
                    AudioEvent(
                        category=SoundCategory.VOCAL_DISTRESS,
                        label="scream",
                        class_index=2,
                        score=0.8,
                        timestamp=2.0,
                    )
                ],
                top_labels=[("scream", 0.8)],
                duration_sec=10.0,
                elapsed_ms=100.0,
            ),
        )
        m.status.current_risk_level = RiskLevel.LOW
        m.status.baseline_ready = True
        m.status.last_feature = None
        m.status.last_deviation = None
        m.status.last_alert = None

        d = m.get_status()
        assert d["audio_enabled"] is True
        assert d["audio_source"] == "mic"
        assert d["audio_chunks_processed"] == 5
        assert d["audio_error"] == "some error"
        assert d["last_audio_result"] is not None
        assert d["last_audio_result"]["events"][0]["category"] == SoundCategory.VOCAL_DISTRESS.value
        assert d["last_audio_result"]["duration_sec"] == 10.0

    def test_no_audio_returns_none_fields(self):
        m = object.__new__(FallRiskMonitor)
        m.status = MonitorStatus()
        m.status.current_risk_level = RiskLevel.LOW
        m.status.baseline_ready = True
        m.status.last_feature = None
        m.status.last_deviation = None
        m.status.last_alert = None

        d = m.get_status()
        assert d["audio_enabled"] is False
        assert d["last_audio_result"] is None


# ── routes 传参 ──


class TestMonitorStartRequestAudioField:
    """MonitorStartRequest 应支持可选的 audio_source"""

    def test_default_none(self):
        from src.api.routes import MonitorStartRequest

        req = MonitorStartRequest(source="0")
        assert req.audio_source is None

    def test_explicit_value(self):
        from src.api.routes import MonitorStartRequest

        req = MonitorStartRequest(source="0", audio_source="mic")
        assert req.audio_source == "mic"
