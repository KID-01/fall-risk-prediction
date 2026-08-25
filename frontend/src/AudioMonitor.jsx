import React, { useState, useEffect, useRef, useCallback } from 'react'
import * as echarts from 'echarts'

const API_BASE = '/api/v1'

export default function AudioMonitor() {
  const [status, setStatus] = useState(null)
  const [results, setResults] = useState(null)
  const [error, setError] = useState('')
  const [uploading, setUploading] = useState(false)
  const [recording, setRecording] = useState(false)
  const [liveAnalyzing, setLiveAnalyzing] = useState(false)
  const [micError, setMicError] = useState('')

  const chartRef = useRef(null)
  const waveformRef = useRef(null)
  const audioContextRef = useRef(null)
  const mediaStreamRef = useRef(null)
  const workletRef = useRef(null)
  const bufferRef = useRef([])
  const recordingStartRef = useRef(0)
  const lastFlushRef = useRef(0)

  // 状态点: idle | ready | recording | analyzing | error
  const statusState = uploading || liveAnalyzing
    ? 'analyzing'
    : recording
      ? 'recording'
      : error
        ? 'error'
        : status?.enabled
          ? 'ready'
          : 'idle'

  // 获取音频状态
  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/audio/status`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setStatus(data)
      setError('')
    } catch (e) {
      setError(`状态获取失败: ${e.message}`)
      setStatus(null)
    }
  }, [])

  useEffect(() => {
    fetchStatus()
  }, [fetchStatus])

  // ECharts Top-k 条形图
  useEffect(() => {
    if (!results?.top_labels?.length || !chartRef.current) return

    const chart = echarts.init(chartRef.current)
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark'
    const textColor = isDark ? '#e2e8f0' : '#1e293b'
    const mutedColor = isDark ? '#64748b' : '#94a3b8'

    const labels = results.top_labels.map(([l]) => l)
    const scores = results.top_labels.map(([, s]) => s)

    chart.setOption({
      tooltip: {
        trigger: 'axis',
        formatter: (params) => `${params[0].name}: ${params[0].value.toFixed(3)}`,
      },
      grid: { left: 10, right: 10, top: 10, bottom: 10 },
      xAxis: {
        type: 'value',
        min: 0,
        max: 1,
        splitLine: { lineStyle: { color: isDark ? '#334155' : '#f1f5f9' } },
        axisLabel: { fontSize: 11, color: mutedColor },
      },
      yAxis: {
        type: 'category',
        data: labels.reverse(),
        axisLine: { lineStyle: { color: isDark ? '#475569' : '#e2e8f0' } },
        axisLabel: { fontSize: 11, color: mutedColor },
        axisTick: { show: false },
      },
      series: [{
        type: 'bar',
        data: scores.reverse(),
        barHeight: '60%',
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
            { offset: 0, color: '#3b82f6' },
            { offset: 1, color: '#60a5fa' },
          ]),
        },
        label: {
          show: true,
          position: 'right',
          formatter: '{c}',
          fontSize: 11,
          color: textColor,
        },
      }],
    })

    return () => chart.dispose()
  }, [results])

  // 文件上传分析
  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    setUploading(true)
    setError('')

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch(`${API_BASE}/audio/analyze`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || `HTTP ${res.status}`)
      }

      const data = await res.json()
      setResults(data)
    } catch (e) {
      setError(`分析失败: ${e.message}`)
    } finally {
      setUploading(false)
    }
  }

  // 绘制波形
  const drawWaveform = useCallback((float32Array) => {
    const canvas = waveformRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const dpr = window.devicePixelRatio || 1
    const width = canvas.clientWidth
    const height = canvas.clientHeight
    canvas.width = width * dpr
    canvas.height = height * dpr
    ctx.scale(dpr, dpr)

    ctx.clearRect(0, 0, width, height)
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--primary') || '#3b82f6'
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

  // 音频工作节点处理
  const handleAudioData = useCallback((float32Array) => {
    bufferRef.current.push(...float32Array)
    drawWaveform(float32Array)

    const now = Date.now()
    const elapsed = (now - recordingStartRef.current) / 1000
    if (elapsed - lastFlushRef.current >= 5 && bufferRef.current.length >= 0.5 * 32000) {
      lastFlushRef.current = elapsed
      flushBuffer(elapsed)
    }
  }, [drawWaveform])

  // 刷新缓冲区并上传
  const flushBuffer = useCallback(async (timestamp) => {
    if (!workletRef.current || !audioContextRef.current) return

    const sampleRate = audioContextRef.current.sampleRate
    const chunk = bufferRef.current.splice(0, bufferRef.current.length)
    if (chunk.length < 0.5 * sampleRate) return

    setLiveAnalyzing(true)
    try {
      // 编码为 WAV
      const { encodeWav } = await import('./audioUtils.js')
      const blob = encodeWav(new Float32Array(chunk), sampleRate)

      const formData = new FormData()
      formData.append('file', blob, `live_${Date.now()}.wav`)

      const res = await fetch(`${API_BASE}/audio/analyze?timestamp=${timestamp}`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setResults(data)
    } catch (e) {
      console.error('实时分析失败:', e)
    } finally {
      setLiveAnalyzing(false)
    }
  }, [])

  // 开始麦克风录音
  const startRecording = async () => {
    setMicError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      mediaStreamRef.current = stream

      const audioContext = new AudioContext({ sampleRate: 32000 })
      audioContextRef.current = audioContext

      // 创建 AudioWorklet (内联 Blob URL)
      const workletCode = `
        class AudioProcessor extends AudioWorkletProcessor {
          constructor() {
            super()
            this.buffer = new Float32Array(1024)
            this.offset = 0
          }
          process(inputs) {
            const input = inputs[0]
            if (input.length > 0) {
              const channel = input[0]
              for (let i = 0; i < channel.length; i++) {
                this.buffer[this.offset++] = channel[i]
                if (this.offset >= 1024) {
                  this.port.postMessage(this.buffer.slice())
                  this.offset = 0
                }
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
      source.connect(worklet).connect(audioContext.destination)

      recordingStartRef.current = Date.now()
      lastFlushRef.current = 0
      setRecording(true)
      setError('')
    } catch (e) {
      setMicError(`麦克风启动失败: ${e.message}`)
      setRecording(false)
    }
  }

  // 停止麦克风录音
  const stopRecording = () => {
    if (workletRef.current) {
      workletRef.current.disconnect()
      workletRef.current = null
    }
    if (audioContextRef.current) {
      audioContextRef.current.close()
      audioContextRef.current = null
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(t => t.stop())
      mediaStreamRef.current = null
    }
    bufferRef.current = []
    setRecording(false)
    setLiveAnalyzing(false)
  }

  // 格式化事件显示
  const formatEvent = (event) => {
    const categoryLabel = event.category === 'vocal_distress' ? '人声呼救' : '撞击声'
    const color = event.category === 'vocal_distress' ? 'var(--orange)' : 'var(--red)'
    return (
      <span key={`${event.class_index}-${event.timestamp}`} className="event-badge" style={{ background: `${color}20`, color, borderColor: color }}>
        [{categoryLabel}] {event.label} ({event.score.toFixed(2)}) @ {event.timestamp.toFixed(1)}s
      </span>
    )
  }

  return (
    <div className="audio-monitor">
      {/* ── 状态栏 ── */}
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
          {status && (
            <span className="audio-status-detail">
              {status.model_type} | {status.sample_rate}Hz | {'模型已加载: ' + (status.model_loaded ? '是' : '否')}
            </span>
          )}
        </div>
        <div className="audio-actions">
          <input
            type="file"
            accept=".wav,.flac,.ogg,audio/*"
            onChange={handleFileUpload}
            disabled={uploading}
            style={{ display: 'none' }}
            id="audio-file-input"
            ref={(el) => { window.audioFileInput = el }}
          />
          <button
            className="btn btn-secondary"
            onClick={() => window.audioFileInput?.click()}
            disabled={uploading || recording}
          >
            {uploading ? '分析中...' : '上传音频文件'}
          </button>
          <button
            className={`btn ${recording ? 'btn-danger' : 'btn-primary'}`}
            onClick={recording ? stopRecording : startRecording}
            disabled={uploading}
          >
            {recording ? '■ 停止录音' : '🎤 开始实时监测'}
          </button>
        </div>
      </div>

      {micError && <div className="audio-error">{micError}</div>}
      {error && <div className="audio-error">{error}</div>}

      {/* ── 配置面板 ── */}
      <div className="audio-config-panel">
        <h4>音频分析配置</h4>
        <div className="config-grid">
          {status && Object.entries(status).map(([key, value]) => (
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
          style={{ width: '100%', height: '120px', background: 'var(--chart-bg)' }}
        />
      </div>

      {/* ── 分析结果 ── */}
      <div className="audio-results">
        {results && (
          <>
            <div className="results-meta">
              <div>时长: {results.duration_sec.toFixed(2)}s</div>
              <div>耗时: {results.elapsed_ms.toFixed(1)}ms</div>
            </div>

            {results.events.length > 0 && (
              <div className="audio-events">
                <h4>检测到的事件 ({results.events.length})</h4>
                <div className="events-list">
                  {results.events.map(formatEvent)}
                </div>
              </div>
            )}

            {results.top_labels.length > 0 && (
              <div className="audio-top-labels">
                <h4>Top 标签</h4>
                <div ref={chartRef} className="audio-chart" style={{ height: '280px' }} />
              </div>
            )}
          </>
        )}
        {!results && !uploading && !liveAnalyzing && (
          <div className="audio-empty">
            <span className="empty-icon">🎧</span>
            <span>上传音频文件或点击「开始实时监测」查看分析结果</span>
          </div>
        )}
      </div>
    </div>
  )
}