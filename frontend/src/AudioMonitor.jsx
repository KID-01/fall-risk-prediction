import React, { useState, useEffect, useRef, useCallback } from 'react'
import * as echarts from 'echarts'

const API_BASE = '/api/v1'

export default function AudioMonitor({ pipelineAudio }) {
  const [status, setStatus] = useState(null)
  const [results, setResults] = useState(null)
  const [error, setError] = useState('')
  const [uploading, setUploading] = useState(false)
  const [recording, setRecording] = useState(false)
  const [liveAnalyzing, setLiveAnalyzing] = useState(false)
  const [micError, setMicError] = useState('')

  const pipelineChartRef = useRef(null)
  const waveformRef = useRef(null)
  const audioContextRef = useRef(null)
  const mediaStreamRef = useRef(null)
  const workletRef = useRef(null)
  const bufferRef = useRef([])
  const recordingStartRef = useRef(0)
  const lastFlushRef = useRef(0)
  const fileInputRef = useRef(null)

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

  // 管线/本地音频 Top-k 图表
  useEffect(() => {
    const activeResult = pipelineAudio?.lastResult || results
    const topLabels = activeResult?.top_labels
    if (!topLabels?.length || !pipelineChartRef.current) return

    const chart = echarts.init(pipelineChartRef.current)
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark'
    const textColor = isDark ? '#e2e8f0' : '#1e293b'
    const mutedColor = isDark ? '#64748b' : '#94a3b8'

    const labels = topLabels.slice(0, 10).map(([l]) => l)
    const scores = topLabels.slice(0, 10).map(([, s]) => s)

    chart.setOption({
      tooltip: {
        trigger: 'axis',
        formatter: (params) => `${params[0].name}: ${params[0].value.toFixed(3)}`,
      },
      grid: { left: 10, right: 40, top: 10, bottom: 10, containLabel: true },
      xAxis: {
        type: 'value',
        min: 0,
        max: 1,
        splitLine: { lineStyle: { color: isDark ? '#334155' : '#f1f5f9' } },
        axisLabel: { fontSize: 11, color: mutedColor },
      },
      yAxis: {
        type: 'category',
        data: [...labels].reverse(),
        axisLabel: { fontSize: 11, color: textColor },
        axisTick: { show: false },
        axisLine: { show: false },
      },
      series: [{
        type: 'bar',
        data: [...scores].reverse(),
        barWidth: 14,
        itemStyle: {
          borderRadius: [0, 4, 4, 0],
          color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
            { offset: 0, color: '#6366f1' },
            { offset: 1, color: '#818cf8' },
          ]),
        },
        label: {
          show: true,
          position: 'right',
          fontSize: 11,
          color: textColor,
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
      if (workletRef.current) {
        workletRef.current.disconnect()
        workletRef.current = null
      }
      if (audioContextRef.current) {
        audioContextRef.current.close()
        audioContextRef.current = null
      }
    }
  }, [])

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
    const sr = audioContextRef.current?.sampleRate || 32000
    if (elapsed - lastFlushRef.current >= 5 && bufferRef.current.length >= 0.5 * sr) {
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
    setError('')
    try {
      const { encodeWav } = await import('./audioUtils.js')
      const blob = encodeWav(new Float32Array(chunk), sampleRate)

      const formData = new FormData()
      formData.append('file', blob, `live_${Date.now()}.wav`)

      const res = await fetch(`${API_BASE}/audio/analyze?timestamp=${timestamp}`, {
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
      setError(`实时分析失败: ${e.message}`)
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
      source.connect(worklet)

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
            ref={fileInputRef}
          />
          <button
            className="btn btn-secondary"
            onClick={() => fileInputRef.current?.click()}
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

      {/* ── 音频分析结果 ── */}
      <div className="pipeline-audio-section">
        <div className="pipeline-audio-header">
          <h4>音频分析结果</h4>
          {pipelineAudio?.enabled && (
            <span className="pipeline-audio-live">
              <span className="live-dot" /> 管线监测中
            </span>
          )}
          {!pipelineAudio?.enabled && results && (
            <span className="pipeline-audio-source-tag">本地分析</span>
          )}
        </div>

        {(() => {
          const activeResult = pipelineAudio?.lastResult || results
          if (!activeResult) {
            return (
              <div className="audio-empty">
                <span className="empty-icon">🎧</span>
                <span>{pipelineAudio?.enabled ? '等待音频数据...' : '上传音频或开始录音查看分析结果'}</span>
              </div>
            )
          }

          return (
            <div className="pipeline-audio-content">
              <div className="pipeline-audio-meta">
                {pipelineAudio?.enabled ? (
                  <div className="pipeline-audio-source">
                    <span className="meta-label">音频来源</span>
                    <span className="meta-value">{pipelineAudio.source || '未知'}</span>
                  </div>
                ) : (
                  <div className="pipeline-audio-source">
                    <span className="meta-label">分析来源</span>
                    <span className="meta-value">本地 {recording ? '麦克风' : '文件上传'}</span>
                  </div>
                )}
                <div className="pipeline-audio-duration">
                  <span className="meta-label">音频时长</span>
                  <span className="meta-value">{activeResult.duration_sec?.toFixed(1)}s</span>
                </div>
                <div className="pipeline-audio-time">
                  <span className="meta-label">分析耗时</span>
                  <span className="meta-value">{activeResult.elapsed_ms?.toFixed(0)}ms</span>
                </div>
                {pipelineAudio?.enabled && (
                  <div className="pipeline-audio-chunks">
                    <span className="meta-label">已处理</span>
                    <span className="meta-value">{pipelineAudio.chunksProcessed || 0} 块</span>
                  </div>
                )}
              </div>

              {pipelineAudio?.error && (
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
                      const catColor = event.category === 'vocal_distress' ? 'var(--orange)' : 'var(--red)'
                      return (
                        <span
                          key={`${event.class_index}-${i}`}
                          className="event-badge"
                          style={{ background: `${catColor}20`, color: catColor, borderColor: catColor }}
                        >
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
    </div>
  )
}