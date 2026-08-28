import React, { useState, useEffect, useRef, useCallback } from 'react'
import * as echarts from 'echarts'
import EzvizPlayer from './EzvizPlayer'

const API_BASE = '/api/v1'

export default function AudioMonitor({
  videoRef, pipelineAudio, isRunning, startMonitor, stopMonitor,
  devices, selectedDeviceId, setSelectedDeviceId, selectedDevice,
  sourceMode, source, channelNo, setChannelNo, devicesLoading,
  audioSource, setAudioSource, playerConfig, setPlayerState, setPlayerError,
}) {
  const [audioStatus, setAudioStatus] = useState(null)
  const [results, setResults] = useState(null)
  const [error, setError] = useState('')
  const [uploading, setUploading] = useState(false)
  const [recording, setRecording] = useState(false)
  const [liveAnalyzing, setLiveAnalyzing] = useState(false)
  const [micError, setMicError] = useState('')
  const [monitorMode, setMonitorMode] = useState('idle')
  const [showEzvizPlayer, setShowEzvizPlayer] = useState(false)

  const pipelineChartRef = useRef(null)
  const waveformRef = useRef(null)
  const audioContextRef = useRef(null)
  const mediaStreamRef = useRef(null)
  const workletRef = useRef(null)
  const bufferRef = useRef([])
  const recordingStartRef = useRef(0)
  const lastFlushRef = useRef(0)
  const fileInputRef = useRef(null)

  // 视频模式自动跟随管线状态
  useEffect(() => {
    if (monitorMode === 'video' && !isRunning) {
      setMonitorMode('idle')
    }
  }, [monitorMode, isRunning])

  // 状态点: idle | ready | recording | analyzing | error
  const statusState = uploading || liveAnalyzing
    ? 'analyzing'
    : recording
      ? 'recording'
      : error || micError
        ? 'error'
        : monitorMode === 'video' && isRunning
          ? 'ready'
          : audioStatus?.model_loaded
            ? 'ready'
            : 'idle'

  // 获取音频状态
  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/audio/status`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setAudioStatus(data)
      setError('')
    } catch (e) {
      setError(`状态获取失败: ${e.message}`)
      setAudioStatus(null)
    }
  }, [])

  useEffect(() => {
    fetchStatus()
  }, [fetchStatus])

  // 管线/本地音频 Top-k 图表
  useEffect(() => {
    const activeResult = pipelineAudio?.lastResult || results
    const topLabels = activeResult?.top_labels
    if (!topLabels?.length || !pipelineChartRef.current) return

    const chart = echarts.init(pipelineChartRef.current)
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark'
    const textColor = isDark ? '#F5F3ED' : '#1A1918'
    const mutedColor = isDark ? '#A29E92' : '#8A887E'

    const labels = topLabels.slice(0, 10).map(([l]) => l)
    const scores = topLabels.slice(0, 10).map(([, s]) => s)

    chart.setOption({
      tooltip: {
        trigger: 'axis',
        formatter: (params) => `${params[0].name}: ${params[0].value.toFixed(3)}`,
      },
      grid: { left: 10, right: 40, top: 10, bottom: 10, containLabel: true },
      xAxis: {
        type: 'value', min: 0, max: 1,
        splitLine: { lineStyle: { color: isDark ? '#2E2C28' : '#EFECE3' } },
        axisLabel: { fontSize: 11, color: mutedColor },
      },
      yAxis: {
        type: 'category', data: [...labels].reverse(),
        axisLabel: { fontSize: 11, color: textColor },
        axisTick: { show: false }, axisLine: { show: false },
      },
      series: [{
        type: 'bar', data: [...scores].reverse(), barWidth: 14,
        itemStyle: {
          borderRadius: [0, 4, 4, 0],
          color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
            { offset: 0, color: '#20808D' }, { offset: 1, color: '#3AA8B5' },
          ]),
        },
        label: {
          show: true, position: 'right', fontSize: 11, color: textColor,
          formatter: (p) => p.value.toFixed(2),
        },
      }],
    })

    const onResize = () => chart.resize()
    window.addEventListener('resize', onResize)
    return () => { chart.dispose(); window.removeEventListener('resize', onResize) }
  }, [pipelineAudio?.lastResult?.top_labels, results?.top_labels])

  // 组件卸载时释放所有音频资源
  useEffect(() => {
    return () => {
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(t => t.stop())
        mediaStreamRef.current = null
      }
      if (workletRef.current) { workletRef.current.disconnect(); workletRef.current = null }
      if (audioContextRef.current) { audioContextRef.current.close(); audioContextRef.current = null }
    }
  }, [])

  // ── 文件上传 ──
  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true); setError('')
    const formData = new FormData()
    formData.append('file', file)
    try {
      const res = await fetch(`${API_BASE}/audio/analyze`, { method: 'POST', body: formData })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || `HTTP ${res.status}`)
      }
      setResults(await res.json())
    } catch (e) { setError(`分析失败: ${e.message}`) }
    finally { setUploading(false) }
  }

  // ── 波形绘制 ──
  const drawWaveform = useCallback((float32Array) => {
    const canvas = waveformRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const dpr = window.devicePixelRatio || 1
    const width = canvas.clientWidth
    const height = canvas.clientHeight
    canvas.width = width * dpr; canvas.height = height * dpr
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, width, height)
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--fr-chart-primary') || '#20808D'
    ctx.globalAlpha = 0.6
    const step = Math.max(1, Math.floor(float32Array.length / width))
    for (let x = 0; x < width; x++) {
      let max = 0
      for (let i = 0; i < step && x * step + i < float32Array.length; i++) {
        max = Math.max(max, Math.abs(float32Array[x * step + i]))
      }
      const h = max * height * 0.4
      ctx.fillRect(x, height / 2 - h / 2, 1, h)
    }
    ctx.globalAlpha = 1
  }, [])

  // ── 音频数据处理 ──
  const handleAudioData = useCallback((float32Array) => {
    bufferRef.current.push(...float32Array)
    drawWaveform(float32Array)
    const now = Date.now()
    const elapsed = (now - recordingStartRef.current) / 1000
    const sr = audioContextRef.current?.sampleRate || 32000
    if (elapsed - lastFlushRef.current >= 5 && bufferRef.current.length >= 0.5 * sr) {
      lastFlushRef.current = elapsed
      flushBuffer(elapsed)
    }
  }, [drawWaveform])

  const flushBuffer = useCallback(async (timestamp) => {
    if (!workletRef.current || !audioContextRef.current) return
    const sampleRate = audioContextRef.current.sampleRate
    const chunk = bufferRef.current.splice(0, bufferRef.current.length)
    if (chunk.length < 0.5 * sampleRate) return
    setLiveAnalyzing(true); setError('')
    try {
      const { encodeWav } = await import('./audioUtils.js')
      const blob = encodeWav(new Float32Array(chunk), sampleRate)
      const formData = new FormData()
      formData.append('file', blob, `live_${Date.now()}.wav`)
      const res = await fetch(`${API_BASE}/audio/analyze?timestamp=${timestamp}`, { method: 'POST', body: formData })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || `HTTP ${res.status}`)
      }
      setResults(await res.json())
    } catch (e) { setError(`实时分析失败: ${e.message}`) }
    finally { setLiveAnalyzing(false) }
  }, [])

  // ── 浏览器麦克风录音 ──
  const startBrowserRecording = async () => {
    setMicError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      mediaStreamRef.current = stream
      const audioContext = new AudioContext({ sampleRate: 32000 })
      audioContextRef.current = audioContext

      const workletCode = `
        class AudioProcessor extends AudioWorkletProcessor {
          constructor() { super(); this.buffer = new Float32Array(1024); this.offset = 0 }
          process(inputs) {
            const input = inputs[0]
            if (input.length > 0) {
              const channel = input[0]
              for (let i = 0; i < channel.length; i++) {
                this.buffer[this.offset++] = channel[i]
                if (this.offset >= 1024) { this.port.postMessage(this.buffer.slice()); this.offset = 0 }
              }
            }
            return true
          }
        }
        registerProcessor('audio-processor', AudioProcessor)
      `
      const blob = new Blob([workletCode], { type: 'application/javascript' })
      const workletUrl = URL.createObjectURL(blob)
      await audioContext.audioWorklet.addModule(workletUrl)
      URL.revokeObjectURL(workletUrl)

      const worklet = new AudioWorkletNode(audioContext, 'audio-processor')
      workletRef.current = worklet
      worklet.port.onmessage = (e) => handleAudioData(e.data)

      const source = audioContext.createMediaStreamSource(stream)
      source.connect(worklet)

      recordingStartRef.current = Date.now()
      lastFlushRef.current = 0
      setRecording(true); setError('')
    } catch (e) {
      setMicError(`麦克风启动失败: ${e.message}`)
      setRecording(false)
    }
  }

  const stopBrowserRecording = () => {
    if (workletRef.current) { workletRef.current.disconnect(); workletRef.current = null }
    if (audioContextRef.current) { audioContextRef.current.close(); audioContextRef.current = null }
    if (mediaStreamRef.current) { mediaStreamRef.current.getTracks().forEach(t => t.stop()); mediaStreamRef.current = null }
    bufferRef.current = []
    setRecording(false); setLiveAnalyzing(false)
  }

  // ── 切换浏览器收音 ──
  const toggleBrowserMode = () => {
    if (monitorMode === 'browser') {
      stopBrowserRecording()
      setMonitorMode('idle')
    } else {
      setMonitorMode('browser')
      startBrowserRecording()
    }
  }

  // ── 切换视频源收音 ──
  const toggleVideoMode = () => {
    if (monitorMode === 'video') {
      setMonitorMode('idle')
      // 不停后端监控，只停止音频
    } else {
      if (!isRunning) {
        // 启动后端监控 (带音频) — 直接传值避免 React state 批量更新延迟
        startMonitor({ stayOnTab: true, audioSource: 'video_source' })
      }
      setMonitorMode('video')
    }
  }

  // 切换模式时，关闭之前的浏览器录音
  useEffect(() => {
    if (monitorMode !== 'browser' && recording) {
      stopBrowserRecording()
    }
  }, [monitorMode])

  // 活跃结果
  const activeResult = monitorMode === 'video' ? pipelineAudio?.lastResult : results
  const isEzvizVideoMode = monitorMode === 'video' && sourceMode === 'ezviz' && playerConfig && isRunning

  return (
    <div className="audio-monitor">
      <div className="audio-video-preview">
        {isEzvizVideoMode ? (
          <EzvizPlayer active={true} config={playerConfig} audio={true} setPlayerState={setPlayerState} setPlayerError={setPlayerError} />
        ) : (
          <img ref={videoRef} alt="监控画面" className="audio-video-thumb" />
        )}
        {monitorMode === 'video' && isRunning && (
          <span className="audio-video-label">
            音频来源: {pipelineAudio?.source || '视频源'}
            {isEzvizVideoMode && ' (含声音)'}
          </span>
        )}
        {monitorMode !== 'video' && !isRunning && (
          <div className="audio-video-placeholder">
            <span>选择「视频源收音」后显示实时画面</span>
          </div>
        )}
      </div>

      {/* ── 视频源设备选择 (仅在视频模式且未运行时显示) ── */}
      {monitorMode === 'video' && !isRunning && (
        <div className="audio-device-select">
          {sourceMode === 'ezviz' ? (
            <div className="audio-device-row">
              <select
                className="select"
                value={selectedDeviceId}
                onChange={e => setSelectedDeviceId(e.target.value)}
              >
                <option value="">{devicesLoading ? '加载中...' : '请选择设备'}</option>
                {devices.map(d => (
                  <option key={d.device_id} value={d.device_id}>
                    {d.name}（{d.display_serial}，{d.online ? '在线' : '离线'}）
                  </option>
                ))}
              </select>
              {selectedDevice && (
                <select
                  className="select audio-channel-select"
                  value={channelNo}
                  onChange={e => setChannelNo(Number(e.target.value))}
                >
                  {(selectedDevice?.channels || []).map(ch => <option key={ch} value={ch}>通道 {ch}</option>)}
                </select>
              )}
            </div>
          ) : (
            <div className="audio-device-row">
              <span className="audio-source-url">{source || '未设置视频源'}</span>
            </div>
          )}
        </div>
      )}

      {/* ── 状态栏 + 操作按钮 ── */}
      <div className="audio-toolbar">
        <div className="audio-status">
          <span className={`audio-status-dot ${statusState}`} data-state={statusState} />
          <span>{{
            idle: '未连接',
            ready: '就绪',
            recording: '录音中...',
            analyzing: '分析中...',
            error: '错误',
          }[statusState]}</span>
          {audioStatus && (
            <span className="audio-status-detail">
              {audioStatus.model_type} | {audioStatus.sample_rate}Hz | 模型{audioStatus.model_loaded ? '已加载' : '加载中'}
            </span>
          )}
          {monitorMode === 'video' && isRunning && (
            <span className="audio-status-detail audio-live-tag">
              <span className="live-dot" /> 管线监测中
            </span>
          )}
        </div>

        <div className="audio-actions">
          <input
            type="file" accept=".wav,.flac,.ogg,audio/*"
            onChange={handleFileUpload}
            disabled={uploading}
            style={{ display: 'none' }}
            ref={fileInputRef}
          />
          <button
            className="btn btn-secondary"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading || recording || monitorMode === 'video'}
          >
            {uploading ? '分析中...' : '上传音频文件'}
          </button>

          {/* 两个监测按钮 */}
          <button
            className={`btn ${monitorMode === 'browser' ? 'btn-danger' : 'btn-primary'}`}
            onClick={toggleBrowserMode}
            disabled={uploading || monitorMode === 'video'}
          >
            {monitorMode === 'browser' ? '■ 停止' : '🎤 浏览器收音'}
          </button>
          <button
            className={`btn ${monitorMode === 'video' ? 'btn-danger' : 'btn-primary'}`}
            onClick={toggleVideoMode}
            disabled={uploading || recording}
          >
            {monitorMode === 'video' ? '■ 停止' : '📹 视频源收音'}
          </button>
        </div>
      </div>

      {micError && <div className="audio-error">{micError}</div>}
      {error && <div className="audio-error">{error}</div>}

      {/* ── 音频分析结果 ── */}
      <div className="pipeline-audio-section">
        <div className="pipeline-audio-header">
          <h4>音频分析结果</h4>
          {monitorMode === 'video' && isRunning && (
            <span className="pipeline-audio-live"><span className="live-dot" /> 视频源音频</span>
          )}
          {monitorMode === 'browser' && (
            <span className="pipeline-audio-source-tag">浏览器麦克风</span>
          )}
          {monitorMode === 'idle' && !activeResult && (
            <span className="pipeline-audio-source-tag">等待分析</span>
          )}
        </div>

        {(() => {
          if (!activeResult) {
            return (
              <div className="audio-empty">
                <span className="empty-icon">🎧</span>
                <span>
                  {monitorMode === 'video' && !isRunning
                    ? '请点击「视频源收音」启动监控'
                    : monitorMode === 'browser'
                      ? '录音后自动上传分析'
                      : '选择监测模式开始分析'}
                </span>
              </div>
            )
          }

          return (
            <div className="pipeline-audio-content">
              <div className="pipeline-audio-meta">
                <div className="pipeline-audio-source">
                  <span className="meta-label">分析来源</span>
                  <span className="meta-value">
                    {monitorMode === 'video' ? `视频源 ${pipelineAudio?.source || ''}` : '浏览器麦克风'}
                  </span>
                </div>
                <div className="pipeline-audio-duration">
                  <span className="meta-label">音频时长</span>
                  <span className="meta-value">{activeResult.duration_sec?.toFixed(1)}s</span>
                </div>
                <div className="pipeline-audio-time">
                  <span className="meta-label">分析耗时</span>
                  <span className="meta-value">{activeResult.elapsed_ms?.toFixed(0)}ms</span>
                </div>
                {monitorMode === 'video' && (
                  <div className="pipeline-audio-chunks">
                    <span className="meta-label">已处理</span>
                    <span className="meta-value">{pipelineAudio?.chunksProcessed || 0} 块</span>
                  </div>
                )}
              </div>

              {pipelineAudio?.error && monitorMode === 'video' && (
                <div className="pipeline-audio-error">
                  <span className="error-icon">⚠️</span> {pipelineAudio.error}
                </div>
              )}

              {activeResult.events?.length > 0 && (
                <div className="pipeline-audio-events">
                  <h5>检测到的声音事件 ({activeResult.events.length})</h5>
                  <div className="events-list">
                    {activeResult.events.map((event, i) => {
                      const catLabel = event.category === 'vocal_distress' ? '人声' : '撞击'
                      const catClass = event.category === 'vocal_distress' ? 'voice' : 'impact'
                      return (
                        <span key={`${event.class_index}-${i}`} className={`event-badge ${catClass}`}>
                          [{catLabel}] {event.label} ({event.score.toFixed(2)})
                        </span>
                      )
                    })}
                  </div>
                </div>
              )}

              {activeResult.top_labels?.length > 0 && (
                <div className="pipeline-audio-topk">
                  <h5>声音标签分布</h5>
                  <div ref={pipelineChartRef} className="audio-chart" style={{ height: '220px' }} />
                  <details className="audio-label-ref">
                    <summary>编号对照表</summary>
                    <div className="label-ref-grid">
                      {[
                        [0, '人声/说话声', 'vocal'], [8, '呼喊', 'vocal'], [10, '欢呼', 'vocal'],
                        [11, '叫喊', 'vocal'], [13, '笑声', 'vocal'], [14, '尖叫', 'vocal'],
                        [22, '哭泣', 'vocal'], [23, '婴儿哭', 'vocal'], [25, '哀号', 'vocal'],
                        [26, '叹气/喘息', 'vocal'], [38, '呻吟', 'vocal'], [44, '喘气', 'vocal'],
                        [46, '脚步声', 'activity'], [47, '跑步声', 'activity'], [66, '拍手', 'activity'],
                        [106, '开关门', 'activity'], [107, '门铃', 'activity'],
                        [358, '摔门/猛击', 'impact'], [359, '敲击', 'impact'],
                        [441, '玻璃碎裂', 'impact'], [443, '破碎', 'impact'],
                        [460, '沉闷撞击', 'impact'], [466, '砰击', 'impact'],
                        [469, '摔砸', 'impact'], [470, '碎裂', 'impact'],
                      ].map(([id, label, cat]) => (
                        <span key={id} className={`label-ref-item label-ref-${cat}`}>
                          <code>{id}</code> {label}
                        </span>
                      ))}
                    </div>
                  </details>
                </div>
              )}
            </div>
          )
        })()}
      </div>

      {/* ── 配置面板 ── */}
      <div className="audio-config-panel">
        <h4>音频分析配置</h4>
        <div className="config-grid">
          {audioStatus && Object.entries(audioStatus).map(([key, value]) => (
            <div key={key} className="config-item">
              <span className="config-label">{key}</span>
              <span className="config-value">{String(value)}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── 实时波形 ── */}
      <div className="audio-waveform-container">
        <h4>实时波形</h4>
        <canvas
          ref={waveformRef}
          className="audio-waveform"
          width={800}
          height={120}
          style={{ width: '100%', height: '120px', background: 'var(--fr-surface)' }}
        />
      </div>
    </div>
  )
}
