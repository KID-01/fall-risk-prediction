import React, { useState, useEffect, useRef, useCallback } from 'react'
import * as echarts from 'echarts'
import AudioMonitor from './AudioMonitor'
import EzvizPlayer from './EzvizPlayer'
import { Shield, Moon, Sun, RefreshCw, Play, Square, RotateCcw, AlertTriangle, MonitorPlay, Video, Bell, ChartLine, AudioLines, Loader, Upload } from './icons'

const API_BASE = '/api/v1'
const LEVEL_LABELS = { low: '低风险', attention: '关注级', warning: '预警级', critical: '高危级' }
const MAX_VIDEO_ASPECT_RATIO = 16 / 9
const MIN_VIDEO_ASPECT_RATIO = 3 / 4
const OBJECT_LABELS = {
  chair: '椅子', couch: '沙发', bed: '床', 'dining table': '餐桌',
  backpack: '背包', suitcase: '行李箱', 'sports ball': '球', laptop: '笔记本电脑',
}

async function readError(response) {
  const payload = await response.json().catch(() => ({}))
  return payload.detail || payload.message || `请求失败（${response.status}）`
}

// 从 CSS 变量读取颜色（双主题自适应）
function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return v || fallback
}

function hexToRgba(hex, alpha) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex.trim())
  if (!m) return hex
  return `rgba(${parseInt(m[1], 16)},${parseInt(m[2], 16)},${parseInt(m[3], 16)},${alpha})`
}

function formatRiskScore(value) {
  const score = Number(value)
  return Number.isFinite(score) ? score.toFixed(1) : '--'
}

function riskScoreClass(state) {
  return `risk-score risk-${String(state || 'unknown').toLowerCase()}`
}

export default function App() {
  const [status, setStatus] = useState({
    is_running: false,
    current_risk_level: 'low',
    current_risk_label: '低风险',
    baseline_ready: false,
    baseline_samples: 0,
    last_feature: null,
    last_alert: null,
    frames_processed: 0,
    frames_valid: 0,
    current_risk_score: 0,
  })
  const [alerts, setAlerts] = useState([])
  const [riskHistory, setRiskHistory] = useState([])
  const [stats, setStats] = useState({})
  const [connected, setConnected] = useState(false)
  const [sourceMode, setSourceMode] = useState('ezviz')
  const [source, setSource] = useState(() => localStorage.getItem('monitor_source') || '')
  const [personId, setPersonId] = useState(() => localStorage.getItem('monitor_person_id') || 'default')
  const [audioSource, setAudioSource] = useState(() => localStorage.getItem('monitor_audio_source') || 'off')
  const [devices, setDevices] = useState([])
  const [selectedDeviceId, setSelectedDeviceId] = useState('')
  const [channelNo, setChannelNo] = useState(1)
  const [devicesLoading, setDevicesLoading] = useState(false)
  const [controlError, setControlError] = useState('')
  const [videoTab, setVideoTab] = useState('analysis')
  const [videoAspectRatio, setVideoAspectRatio] = useState(16 / 9)
  const [environmentPanelPercent, setEnvironmentPanelPercent] = useState(24)
  const resizingEnvironmentRef = useRef(false)
  const [playerConfig, setPlayerConfig] = useState(null)
  const [rawPlayerLoaded, setRawPlayerLoaded] = useState(false)
  const [playerState, setPlayerState] = useState('idle')
  const [playerError, setPlayerError] = useState('')
  const [developerUploading, setDeveloperUploading] = useState(false)
  const [developerStarting, setDeveloperStarting] = useState(false)
  const [developerUpload, setDeveloperUpload] = useState(null)
  const [developerVideoName, setDeveloperVideoName] = useState('')
  const [developerVideoActive, setDeveloperVideoActive] = useState(false)
  const developerVideoInputRef = useRef(null)
  const wasRunningRef = useRef(false)

  useEffect(() => {
    const move = (event) => {
      if (!resizingEnvironmentRef.current) return
      const wrap = document.querySelector('.video-frame-wrap')
      if (!wrap) return
      const rect = wrap.getBoundingClientRect()
      const rightPercent = ((rect.right - event.clientX) / rect.width) * 100
      setEnvironmentPanelPercent(Math.min(50, Math.max(20, rightPercent)))
    }
    const stop = () => { resizingEnvironmentRef.current = false; document.body.style.cursor = '' }
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', stop)
    return () => { window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', stop) }
  }, [])
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('theme')
    return saved || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
  })
  const wsRef = useRef(null)
  const videoWsRef = useRef(null)
  const videoImgRef = useRef(null)
  const videoAudioRef = useRef(null)
  const gaugeRef = useRef(null)
  const trendRef = useRef(null)

  const selectedDevice = devices.find(device => device.device_id === selectedDeviceId)

  const handleVideoImageLoad = useCallback((event) => {
    const { naturalWidth, naturalHeight } = event.currentTarget
    if (!naturalWidth || !naturalHeight) return
    const sourceRatio = naturalWidth / naturalHeight
    const nextRatio = Math.min(MAX_VIDEO_ASPECT_RATIO, Math.max(MIN_VIDEO_ASPECT_RATIO, sourceRatio))
    setVideoAspectRatio(current => Math.abs(current - nextRatio) > 0.01 ? nextRatio : current)
  }, [])

  const fetchEzvizDevices = useCallback(async () => {
    setDevicesLoading(true)
    setControlError('')
    try {
      const response = await fetch(`${API_BASE}/ezviz/devices`)
      if (!response.ok) throw new Error(await readError(response))
      const nextDevices = (await response.json()).devices || []
      setDevices(nextDevices)
      setSelectedDeviceId(current => {
        if (current && nextDevices.some(device => device.device_id === current)) return current
        return (nextDevices.find(device => device.online) || nextDevices[0])?.device_id || ''
      })
    } catch (error) {
      setDevices([])
      setSelectedDeviceId('')
      setControlError(error.message || '萤石设备列表加载失败')
    } finally {
      setDevicesLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchEzvizDevices()
  }, [fetchEzvizDevices])

  useEffect(() => {
    if (selectedDevice && !selectedDevice.channels.includes(Number(channelNo))) {
      setChannelNo(selectedDevice.channels[0])
    }
  }, [selectedDevice, channelNo])

  const changeSourceMode = mode => {
    if (status.is_running) return
    setSourceMode(mode)
    setControlError('')
    setPlayerConfig(null)
    setPlayerError('')
    setVideoTab('analysis')
  }

  // ── 主题切换 ──
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  const toggleTheme = () => setTheme(prev => prev === 'light' ? 'dark' : 'light')

  // ── 获取数据 ──
  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/risk/current`)
      const data = await res.json()
      setStatus(data)
    } catch (e) { /* 后端未启动时静默 */ }
  }, [])

  const fetchAlerts = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/alerts?limit=10`)
      const data = await res.json()
      setAlerts(data.alerts || [])
    } catch (e) { /* 静默 */ }
  }, [])

  const fetchRiskHistory = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/risk/history?hours=24&limit=100`)
      const data = await res.json()
      setRiskHistory(data.records || [])
    } catch (e) { /* 静默 */ }
  }, [])

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/stats?hours=24`)
      const data = await res.json()
      setStats(data)
    } catch (e) { /* 静默 */ }
  }, [])

  // ── WebSocket ──
  useEffect(() => {
    const wsUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/alerts`
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => setConnected(true)
    ws.onclose = () => setConnected(false)
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data)
      if (msg.type === 'alert') { fetchAlerts(); fetchStatus(); fetchRiskHistory() }
      if (msg.type === 'risk_update') fetchStatus()
    }

    return () => ws.close()
  }, [])

  // ── 视频 WebSocket (分析画面 — 骨骼叠加) ──
  useEffect(() => {
    if (videoTab !== 'analysis') return undefined

    setVideoAspectRatio(16 / 9)

    let stopped = false
    let currentObjectUrl = ''
    const connectVideo = () => {
      if (stopped) return
      const wsUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/video`
      const ws = new WebSocket(wsUrl)
      videoWsRef.current = ws

      ws.onmessage = (event) => {
        if (event.data instanceof Blob) {
          const url = URL.createObjectURL(event.data)
          if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl)
          currentObjectUrl = url
          if (videoImgRef.current) videoImgRef.current.src = url
        }
      }

      ws.onclose = () => {
        if (!stopped) setTimeout(connectVideo, 3000)
      }
    }
    connectVideo()
    return () => {
      stopped = true
      videoWsRef.current?.close()
      if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl)
    }
  }, [videoTab])

  // ── 原始视频 WebSocket (声音监测 — 无骨骼) ──
  useEffect(() => {
    if (videoTab !== 'audio') return undefined

    let stopped = false
    let currentObjectUrl = ''
    const connectRaw = () => {
      if (stopped) return
      const wsUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/video/raw`
      const ws = new WebSocket(wsUrl)
      videoWsRef.current = ws

      ws.onmessage = (event) => {
        if (event.data instanceof Blob) {
          const url = URL.createObjectURL(event.data)
          if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl)
          currentObjectUrl = url
          if (videoAudioRef.current) videoAudioRef.current.src = url
        }
      }

      ws.onclose = () => {
        if (!stopped) setTimeout(connectRaw, 3000)
      }
    }
    connectRaw()
    return () => {
      stopped = true
      videoWsRef.current?.close()
      if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl)
    }
  }, [videoTab])

  // ── 定时刷新 ──
  useEffect(() => {
    fetchStatus(); fetchAlerts(); fetchRiskHistory(); fetchStats()
    const interval = setInterval(() => {
      fetchStatus(); fetchAlerts(); fetchRiskHistory(); fetchStats()
    }, 5000)
    return () => clearInterval(interval)
  }, [])

  // ── 仪表盘 ──
  useEffect(() => {
    if (!gaugeRef.current) return
    const chart = echarts.init(gaugeRef.current)
    const textColor = cssVar('--fr-foreground', '#0f172a')
    const labelColor = cssVar('--fr-chart-text', '#64748b')
    const tickColor = cssVar('--fr-chart-text', '#94a3b8')
    const primary = cssVar('--fr-chart-primary', '#1d4ed8')
    const latestHistoryScore = Number(riskHistory[0]?.risk_score || 0)
    const score = Number(status.current_risk_score ?? latestHistoryScore)
    chart.setOption({
      series: [{
        type: 'gauge',
        radius: '85%',
        min: 0, max: 100,
        startAngle: 210, endAngle: -30,
        axisLine: {
          lineStyle: {
            width: 18,
            color: [
              [0.3, cssVar('--fr-chart-low', '#15803d')],
              [0.5, cssVar('--fr-chart-attention', '#a16207')],
              [0.75, cssVar('--fr-chart-warning', '#c2410c')],
              [1, cssVar('--fr-chart-critical', '#b91c1c')],
            ],
          },
        },
        pointer: { width: 5, itemStyle: { color: primary } },
        axisTick: { distance: -18, length: 6, lineStyle: { width: 1, color: tickColor } },
        splitLine: { distance: -22, length: 14, lineStyle: { width: 2, color: tickColor } },
        axisLabel: { distance: 36, fontSize: 13, fontWeight: 600, color: labelColor },
        anchor: { show: true, size: 14, itemStyle: { color: primary } },
        title: { offsetCenter: [0, '78%'], fontSize: 14, color: labelColor },
        detail: {
          valueAnimation: true,
          formatter: value => Number(value).toFixed(1),
          fontSize: 36,
          fontWeight: 700,
          color: textColor,
          offsetCenter: [0, '55%'],
        },
        data: [{ value: score, name: '风险评分' }],
      }],
    })
    return () => chart.dispose()
  }, [status, riskHistory, theme])

  // ── 趋势图 ──
  useEffect(() => {
    if (!trendRef.current) return
    const chart = echarts.init(trendRef.current)
    if (riskHistory.length === 0) {
      chart.setOption({})
      return
    }
    const textColor = cssVar('--fr-foreground', '#0f172a')
    const mutedColor = cssVar('--fr-chart-text', '#94a3b8')
    const axisColor = cssVar('--fr-border', '#e2e8f0')
    const splitColor = cssVar('--fr-chart-grid', '#f1f5f9')
    const tooltipBg = cssVar('--fr-card', '#ffffff')
    const tooltipBorder = cssVar('--fr-border', '#e2e8f0')
    const primary = cssVar('--fr-chart-primary', '#1d4ed8')
    const times = riskHistory.map(r => new Date(r.timestamp * 1000).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })).reverse()
    const scores = riskHistory.map(r => Math.min(100, Math.max(0, Number(r.risk_score) || 0))).reverse()
    const dataMin = Math.min(...scores)
    const dataMax = Math.max(...scores)
    const dataSpan = Math.max(dataMax - dataMin, 10)
    const padding = Math.max(dataSpan * 0.2, 5)
    const yMin = Math.max(0, Math.floor((dataMin - padding) / 5) * 5)
    const yMax = Math.min(100, Math.ceil((dataMax + padding) / 5) * 5)
    chart.setOption({
      tooltip: {
        trigger: 'axis',
        backgroundColor: tooltipBg,
        borderColor: tooltipBorder,
        textStyle: { color: textColor, fontSize: 13 },
        boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
      },
      grid: { left: 8, right: 16, top: 8, bottom: 8 },
      xAxis: {
        type: 'category', data: times,
        axisLine: { lineStyle: { color: axisColor } },
        axisLabel: { fontSize: 11, color: mutedColor },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value', min: yMin, max: yMax,
        splitLine: { lineStyle: { color: splitColor } },
        axisLabel: { fontSize: 11, color: mutedColor },
      },
      series: [{
        type: 'line', data: scores, smooth: true,
        symbol: 'circle', symbolSize: 4,
        lineStyle: { width: 3, color: primary },
        itemStyle: { color: primary },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: hexToRgba(primary, 0.15) },
            { offset: 1, color: hexToRgba(primary, 0.01) },
          ]),
        },
        markLine: {
          silent: true,
          symbol: 'none',
          label: { show: false },
          lineStyle: { type: 'dashed', width: 1, color: mutedColor, opacity: 0.65 },
          data: [30, 50, 75]
            .filter(value => value >= yMin && value <= yMax)
            .map(value => ({ yAxis: value })),
        },
      }],
    })
    return () => chart.dispose()
  }, [riskHistory, theme])

  // ── 控制操作 ──
  const startMonitor = async ({ stayOnTab, audioSource: audioSourceOverride } = {}) => {
    const effectiveAudioSource = audioSourceOverride || audioSource
    localStorage.setItem('monitor_person_id', personId)
    localStorage.setItem('monitor_audio_source', effectiveAudioSource)
    setControlError('')
    setPlayerError('')

    try {
      let response
      if (sourceMode === 'ezviz') {
        if (!selectedDevice) {
          setControlError('请先选择萤石设备')
          return
        }
        if (!selectedDevice.online) {
          setControlError('所选设备当前离线')
          return
        }
        response = await fetch(`${API_BASE}/ezviz/monitor/start`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            device_id: selectedDeviceId,
            channel_no: Number(channelNo),
            person_id: personId,
            audio_source: effectiveAudioSource,
          }),
        })
      } else {
        if (!source.trim()) {
          setControlError('请输入本地文件、RTMP 或 RTSP 视频源地址')
          return
        }
        localStorage.setItem('monitor_source', source)
        response = await fetch(`${API_BASE}/stream/start`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ source, person_id: personId, audio_source: effectiveAudioSource }),
        })
      }

      if (!response.ok) {
        setControlError(await readError(response))
        return
      }
      if (sourceMode === 'ezviz') setPlayerConfig(await response.json())
      else setPlayerConfig(null)
      if (!stayOnTab && videoTab !== 'analysis') setVideoTab('analysis')
      fetchStatus()
    } catch (_) {
      setControlError('无法连接后端服务，请确认 FastAPI 已启动')
    }
  }
  const stopMonitor = async () => {
    try {
      const response = await fetch(`${API_BASE}/stream/stop`, { method: 'POST' })
      if (!response.ok) {
        setControlError(await readError(response))
        return
      }
      setPlayerConfig(null)
      setRawPlayerLoaded(false)
      setPlayerError('')
      setPlayerState('idle')
      setVideoTab('analysis')
      setDeveloperVideoName('')
      setDeveloperVideoActive(false)
      fetchStatus()
    } catch (_) {
      setControlError('无法连接后端服务，请确认 FastAPI 已启动')
    }
  }
  const uploadDeveloperVideo = async event => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    setDeveloperUploading(true)
    setControlError('')
    setPlayerError('')
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('person_id', personId || 'default')
      form.append('device_id', selectedDeviceId || 'default')
      form.append('audio_source', audioSource || 'off')
      const response = await fetch(`${API_BASE}/stream/upload`, { method: 'POST', body: form })
      if (!response.ok) {
        setControlError(await readError(response))
        return
      }
      const payload = await response.json()
      setDeveloperUpload({ ...payload, source_name: payload.source_name || file.name })
    } catch (_) {
      setControlError('无法上传视频，请确认后端服务已启动')
    } finally {
      setDeveloperUploading(false)
    }
  }

  const cancelDeveloperUpload = async () => {
    const uploadId = developerUpload?.upload_id
    setDeveloperUpload(null)
    if (!uploadId) return
    try {
      const response = await fetch(`${API_BASE}/stream/upload/${encodeURIComponent(uploadId)}`, { method: 'DELETE' })
      if (!response.ok && response.status !== 404) setControlError(await readError(response))
    } catch (_) {
      setControlError('无法取消暂存视频，请确认后端服务已启动')
    }
  }

  const startDeveloperAnalysis = async () => {
    if (!developerUpload?.upload_id || developerStarting) return
    setDeveloperStarting(true)
    setControlError('')
    try {
      const response = await fetch(`${API_BASE}/stream/upload/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          upload_id: developerUpload.upload_id,
          person_id: personId || 'default',
          device_id: selectedDeviceId || 'default',
          audio_source: audioSource || 'off',
        }),
      })
      if (!response.ok) {
        setControlError(await readError(response))
        return
      }
      const payload = await response.json()
      setDeveloperUpload(null)
      setDeveloperVideoName(payload.source_name || developerUpload.source_name)
      setDeveloperVideoActive(true)
      setPlayerConfig(null)
      setRawPlayerLoaded(false)
      setVideoAspectRatio(MAX_VIDEO_ASPECT_RATIO)
      setVideoTab('analysis')
      fetchStatus()
    } catch (_) {
      setControlError('无法开始视频分析，请确认后端服务已启动')
    } finally {
      setDeveloperStarting(false)
    }
  }

  useEffect(() => {
    if (status.is_running) {
      wasRunningRef.current = true
    } else if (wasRunningRef.current) {
      wasRunningRef.current = false
      setDeveloperVideoName('')
      setDeveloperVideoActive(false)
    }
  }, [status.is_running])
  const refreshPlayer = async () => {
    if (!selectedDeviceId) return
    setPlayerState('loading')
    setPlayerError('')
    try {
      const response = await fetch(`${API_BASE}/ezviz/player`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ device_id: selectedDeviceId, channel_no: Number(channelNo) }),
      })
      if (!response.ok) {
        setPlayerState('error')
        setPlayerError(await readError(response))
        return
      }
      setPlayerConfig(await response.json())
      setRawPlayerLoaded(true)
    } catch (_) {
      setPlayerState('error')
      setPlayerError('无法连接后端服务，请确认 FastAPI 已启动')
    }
  }
  const resetBaseline = async () => {
    await fetch(`${API_BASE}/baseline/reset`, { method: 'POST' })
    fetchStatus()
  }

  const level = status.current_risk_level || 'low'
  const levelLabel = LEVEL_LABELS[level] || '低风险'

  // 视频徽标状态
  const videoBadgeState = videoTab === 'raw'
    ? (playerState === 'playing' ? { dot: 'online', text: '正在播放', live: true }
      : playerState === 'loading' ? { dot: 'analyzing', text: '正在连接', live: false }
        : playerState === 'error' ? { dot: 'offline', text: '播放失败', live: false }
          : { dot: 'offline', text: '待启动', live: false })
    : videoTab === 'audio'
      ? { dot: status.audio_enabled ? 'online' : 'offline', text: '声音监测', live: !!status.audio_enabled }
      : (status.is_running ? { dot: 'online', text: '监控中', live: true } : { dot: 'offline', text: '待启动', live: false })

  return (
    <div className="dashboard">
      {/* ── 顶部系统栏 ── */}
      <header className="topbar">
        <div className="topbar-brand">
          <span className="topbar-brand-icon"><Shield size={20} /></span>
          <div>
            <div className="topbar-title">跌倒风险预测系统</div>
            <div className="topbar-sub">基于多模态 AI 监测 · 家属端实时看板</div>
          </div>
        </div>
        <div className="topbar-right">
          <button className="btn btn-ghost btn-icon" onClick={toggleTheme} title={theme === 'light' ? '切换暗色模式' : '切换亮色模式'} aria-label="切换主题">
            {theme === 'light' ? <Moon size={16} /> : <Sun size={16} />}
          </button>
          <span className={`badge ${connected ? 'badge-live' : 'badge-idle'}`}>
            <span className={`status-dot ${connected ? 'online' : 'offline'}`} />
            {connected ? '实时连接' : '未连接'}
          </span>
        </div>
      </header>

      {/* 页面导语保持紧凑，确保监控控制与风险状态仍在首屏可见 */}
      <section className="page-hero" aria-label="页面导语">
        <div className="page-hero-content">
          <span className="page-hero-label">多模态 AI 实时监测</span>
          <h1 className="page-hero-title">跌倒风险实时看板</h1>
          <p className="page-hero-sub">视频、音频与环境三模态融合分析，持续守护被监测人的每一次日常活动。</p>
        </div>
      </section>

      {/* ── 监控控制区 ── */}
      <section className="card controls-card" aria-label="监控控制">
        <div className="card-head">
          <h2 className="section-title">监控设置</h2>
          {status.is_running && <span className="badge badge-live"><span className="status-dot online" />监控中</span>}
        </div>

        <div className="seg" role="group" aria-label="视频源模式">
          <button type="button" disabled={status.is_running} className={sourceMode === 'ezviz' ? 'active' : ''} onClick={() => changeSourceMode('ezviz')}>萤石设备</button>
          <button type="button" disabled={status.is_running} className={sourceMode === 'manual' ? 'active' : ''} onClick={() => changeSourceMode('manual')}>手工地址</button>
        </div>

        <div className="form-grid">
          {sourceMode === 'ezviz' ? (
            <>
              <div className="field-group">
                <label className="field-label" htmlFor="device-select">萤石设备</label>
                <select id="device-select" className="select" value={selectedDeviceId} onChange={e => setSelectedDeviceId(e.target.value)} aria-label="萤石设备" disabled={status.is_running}>
                  <option value="">{devicesLoading ? '正在加载设备' : '请选择设备'}</option>
                  {devices.map(device => (
                    <option key={device.device_id} value={device.device_id}>
                      {device.name}（{device.display_serial}，{device.online ? '在线' : '离线'}）
                    </option>
                  ))}
                </select>
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="channel-select">设备通道</label>
                <select id="channel-select" className="select" value={channelNo} onChange={e => setChannelNo(Number(e.target.value))} disabled={!selectedDevice || status.is_running} aria-label="设备通道">
                  {(selectedDevice?.channels || []).map(channel => <option key={channel} value={channel}>通道 {channel}</option>)}
                </select>
              </div>
              <div className="field-group field-action">
                <label className="field-label">&nbsp;</label>
                <button className="btn btn-secondary" type="button" onClick={fetchEzvizDevices} disabled={devicesLoading || status.is_running}>
                  <RefreshCw size={14} /> {devicesLoading ? '加载中' : '刷新设备'}
                </button>
              </div>
            </>
          ) : (
            <div className="field-group field-wide">
              <label className="field-label" htmlFor="source-url">视频源地址</label>
              <input
                id="source-url"
                className="input"
                type="text"
                placeholder="本地文件、RTMP 或 RTSP 视频源地址"
                value={source}
                onChange={e => setSource(e.target.value)}
                disabled={status.is_running}
              />
            </div>
          )}
        </div>

        <div className="form-grid">
          <div className="field-group">
            <label className="field-label" htmlFor="person-id">被监测人 ID</label>
            <input
              id="person-id"
              className="input"
              type="text"
              placeholder="被监测人 ID"
              value={personId}
              onChange={e => setPersonId(e.target.value)}
              disabled={status.is_running}
            />
          </div>
          <div className="field-group">
            <label className="field-label" htmlFor="audio-source">音频源</label>
            <select
              id="audio-source"
              className="select"
              value={audioSource}
              onChange={e => setAudioSource(e.target.value)}
              disabled={status.is_running}
              aria-label="音频源"
            >
              <option value="auto">跟随视频源(推荐)</option>
              <option value="off">关闭音频</option>
            </select>
          </div>
        </div>

        <div className="control-buttons">
          <button className="btn btn-primary" onClick={startMonitor} disabled={status.is_running || (sourceMode === 'ezviz' && (!selectedDevice || !selectedDevice.online))}>
            <Play size={14} /> 启动监控
          </button>
          <button className="btn btn-danger" onClick={stopMonitor} disabled={!status.is_running}>
            <Square size={14} /> 停止监控
          </button>
          <button className="btn btn-secondary" onClick={resetBaseline}>
            <RotateCcw size={14} /> 重置基线
          </button>
        </div>

        {controlError && (
          <div className="error-banner control-error" role="alert">
            <AlertTriangle size={14} /> {controlError}
          </div>
        )}
      </section>

      {/* ── 视频画面 ── */}
      <section className="card video-panel" aria-label="实时画面">
        <div className="card-head">
          <h2 className="section-title">实时画面</h2>
          <div className="video-head-right">
            <div className="seg" role="tablist" aria-label="视频画面">
              <button type="button" role="tab" aria-selected={videoTab === 'analysis'} className={videoTab === 'analysis' ? 'active' : ''} onClick={() => setVideoTab('analysis')}>AI 分析画面</button>
              <button type="button" role="tab" aria-selected={videoTab === 'raw'} className={videoTab === 'raw' ? 'active' : ''} onClick={() => setVideoTab('raw')} disabled={sourceMode !== 'ezviz' || developerVideoActive}>萤石原始画面</button>
              <button type="button" role="tab" aria-selected={videoTab === 'audio'} className={videoTab === 'audio' ? 'active' : ''} onClick={() => setVideoTab('audio')}>
                <AudioLines size={13} /> 声音监测
              </button>
            </div>
            <span className={`badge ${videoBadgeState.live ? 'badge-live' : 'badge-idle'}`}>
              <span className={`status-dot ${videoBadgeState.dot}`} />
              {videoBadgeState.text}
            </span>
          </div>
        </div>
        <div
          className="video-frame-wrap"
          style={videoTab === 'analysis'
            ? {
                '--environment-width': `${environmentPanelPercent}%`,
                '--video-aspect-ratio': videoAspectRatio,
              }
            : undefined}
        >
          {videoTab === 'analysis' ? (
            <>
              <div className="analysis-video-frame">
                <img ref={videoImgRef} onLoad={handleVideoImageLoad} alt="AI 分析实时画面" className="video-frame" />
              </div>
              <div className="environment-resize-handle" role="separator" aria-label="调整环境检测栏宽度" onMouseDown={() => { resizingEnvironmentRef.current = true; document.body.style.cursor = 'col-resize' }} />
              <div className="developer-tool-float">
                <button
                  type="button"
                  className="developer-tool-button"
                  title="开发者工具"
                  aria-label="开发者工具"
                  onClick={() => developerVideoInputRef.current?.click()}
                  disabled={developerUploading || developerStarting}
                >
                  {developerUploading ? <Loader size={16} className="spin" /> : <Upload size={16} />}
                </button>
                <input
                  ref={developerVideoInputRef}
                  className="developer-video-input"
                  type="file"
                  accept=".mp4,.avi,.mov,.mkv,.webm"
                  onChange={uploadDeveloperVideo}
                />
                {developerUpload && (
                  <div className="developer-upload-confirm" role="status">
                    <strong title={developerUpload.source_name}>待分析：{developerUpload.source_name}</strong>
                    <div className="developer-upload-actions">
                      <button type="button" className="btn btn-primary btn-sm" onClick={startDeveloperAnalysis} disabled={developerStarting}>
                        {developerStarting ? <Loader size={13} className="spin" /> : <Play size={13} />} 开始分析
                      </button>
                      <button type="button" className="btn btn-ghost btn-sm" onClick={cancelDeveloperUpload} disabled={developerStarting}>取消</button>
                    </div>
                  </div>
                )}
              </div>
              <aside className="analysis-environment-panel">
                <div className="environment-model-status">
                  Pose {status.pose_model_loaded ? '已加载' : '待加载'} · 环境 {status.environment_model_loaded ? '已加载' : '待加载'} · {status.is_running ? '实时监测' : '未启动'}
                </div>
                <div className="environment-risk-summary">
                  <div className="environment-risk-metric">
                    <span>综合环境风险</span>
                    <b className={riskScoreClass(status.environment?.state)}>{formatRiskScore(status.environment?.risk_index)}</b>
                  </div>
                  <div className="environment-risk-metric">
                    <span>交互风险</span>
                    <b className={riskScoreClass(status.interaction?.state)}>{formatRiskScore(status.interaction?.risk_index)}</b>
                  </div>
                </div>
                <h3>环境检测</h3>
                <div className="environment-extension-grid">
                  <span>照明 <b>{status.low_light?.state || '--'}</b></span>
                  <span>障碍物 <b>{status.obstacle?.state || '--'}</b></span>
                  <span>轨迹 <b>{status.trajectory?.state || '--'}</b></span>
                  <span>交互 <b>{status.interaction?.state || '--'}</b></span>
                </div>
                <div>识别目标: {status.environment_count || 0}</div>
                {status.environment_error && <div className="text-error">模型不可用</div>}
                {status.environment?.stale && <div className="text-error">环境结果已过期</div>}
                {!status.environment_error && !(status.environment_boxes || []).length && <div>暂未识别到环境目标</div>}
                {(status.environment_boxes || []).map((box, index) => (
                  <div className="environment-item" key={`${box.label}-${index}`}>
                    <span>{OBJECT_LABELS[box.label] || box.label}</span><strong>{(box.confidence * 100).toFixed(1)}%</strong>
                  </div>
                ))}
                {(status.top_hazards || []).length > 0 && <h3>主要危险物</h3>}
                {(status.top_hazards || []).map((hazard, index) => (
                  <div className="environment-hazard" key={`${hazard.label || hazard.class}-${index}`}>
                    <span>{OBJECT_LABELS[hazard.label || hazard.class] || hazard.label || hazard.class}</span>
                    <strong>{(hazard.risk_contribution * 100).toFixed(1)}</strong>
                    <small>距离 {hazard.normalized_distance == null ? '--' : Number(hazard.normalized_distance).toFixed(2)}×身高</small>
                  </div>
                ))}
                <div>光照亮度: {status.illumination == null ? '--' : `${Math.round(status.illumination)}/255`}</div>
                {developerVideoName && developerVideoActive && <div className="developer-video-name" title={developerVideoName}>开发者视频：{developerVideoName}</div>}
              </aside>
              {status.is_running && (
                <div className="video-detection-summary">
                  <span>人体: {status.human_detected ? `已检测 (${status.person?.candidate_count || 1})` : '未检测'}</span>
                  <span>环境目标: {status.environment_count || 0}</span>
                  <span>光照: {status.illumination == null ? '--' : `${Math.round(status.illumination)}/255`}</span>
                  {status.environment_error && <span className="text-error">环境检测不可用</span>}
                </div>
              )}
              {!status.is_running && (
                <div className="video-placeholder">
                  <MonitorPlay size={30} />
                  <span>点击「启动监控」开始查看分析画面</span>
                </div>
              )}
            </>
          ) : videoTab === 'raw' ? (
            <>
              {rawPlayerLoaded && playerConfig
                ? <EzvizPlayer active={videoTab === 'raw'} config={playerConfig} setPlayerState={setPlayerState} setPlayerError={setPlayerError} />
                : <div className="video-placeholder"><Video size={30} /><span>选择在线设备并启动监控后显示原始画面</span></div>}
              {playerState === 'error' && (
                <div className="video-error" role="alert"><AlertTriangle size={14} /> {playerError}</div>
              )}
            </>
          ) : (
            <AudioMonitor
              videoRef={videoAudioRef}
              pipelineAudio={{
                enabled: status.audio_enabled,
                source: status.audio_source,
                chunksProcessed: status.audio_chunks_processed,
                error: status.audio_error || (
                  audioSource === 'video_source' && status.is_running && !status.audio_enabled
                    ? (audioStatus?.checkpoint_exists
                      ? '视频源音频未启用，请停止监控后重新点击“视频源收音”'
                      : 'PANNs 音频模型未安装，无法进行音频分析')
                    : ''
                ),
                lastResult: status.last_audio_result,
                lastAlert: status.last_alert,
              }}
              isRunning={status.is_running}
              startMonitor={startMonitor}
              stopMonitor={stopMonitor}
              devices={devices}
              selectedDeviceId={selectedDeviceId}
              setSelectedDeviceId={setSelectedDeviceId}
              selectedDevice={selectedDevice}
              sourceMode={sourceMode}
              source={source}
              channelNo={channelNo}
              setChannelNo={setChannelNo}
              devicesLoading={devicesLoading}
              audioSource={audioSource}
              setAudioSource={setAudioSource}
              playerConfig={playerConfig}
              setPlayerState={setPlayerState}
              setPlayerError={setPlayerError}
            />
          )}
        </div>
        {videoTab === 'raw' && selectedDeviceId && (
          <div className="video-actions">
            <button className="btn btn-secondary" type="button" onClick={refreshPlayer}>
              <RefreshCw size={14} /> {playerConfig ? '刷新播放授权' : '加载原始画面'}
            </button>
          </div>
        )}
      </section>

      {/* ── 风险等级大卡片 ── */}
      <section className={`risk-hero risk-${level}`} aria-label="当前风险状态">
        <div className="risk-hero-main">
          <div className="risk-level">
            <span className="risk-dot" />
            <span className="risk-level-label">{levelLabel}</span>
          </div>
          <div className="risk-message">
            {status.current_risk_message || status.last_alert?.message || '系统运行正常，持续监测中'}
          </div>
        </div>
        <div className="risk-meta">
          <div className="meta-item">
            <div className="meta-value mono">{status.baseline_ready ? '✓' : `${status.baseline_samples || 0}/100`}</div>
            <div className="meta-label">基线采集</div>
          </div>
          <div className="meta-item">
            <div className="meta-value mono">{status.frames_processed || 0}</div>
            <div className="meta-label">处理帧数</div>
          </div>
          <div className="meta-item">
            <div className="meta-value mono">{status.frames_valid || 0}</div>
            <div className="meta-label">有效帧数</div>
          </div>
          <div className="meta-item">
            <div className="meta-value mono">{status.audio_enabled ? (status.audio_chunks_processed || 0) : '—'}</div>
            <div className="meta-label">音频块数</div>
          </div>
        </div>
        {status.audio_error && (
          <div className="error-banner audio-error-banner" role="alert">
            <AlertTriangle size={14} /> 音频异常: {status.audio_error}
          </div>
        )}
      </section>

      {/* ── 图表网格 ── */}
      <div className="chart-grid">
        <section className="card chart-card" aria-label="当前风险评分">
          <div className="card-head">
            <h2 className="section-title">当前风险评分</h2>
          </div>
          <div ref={gaugeRef} className="chart-container" />
        </section>
        <section className="card chart-card" aria-label="近24小时风险趋势">
          <div className="card-head">
            <h2 className="section-title">近24小时风险趋势</h2>
          </div>
          {riskHistory.length === 0 ? (
            <div className="empty-state chart-empty">
              <ChartLine size={32} />
              <div className="empty-title">暂无历史数据</div>
              <div className="empty-hint">启动监控后数据将在此展示</div>
            </div>
          ) : (
            <div ref={trendRef} className="chart-container" />
          )}
        </section>
      </div>

      {/* ── 告警列表 ── */}
      <section className="card alert-card" aria-label="最新告警">
        <div className="card-head">
          <h2 className="section-title">最新告警</h2>
          <span className="badge badge-neutral">近 24 小时</span>
        </div>
        {alerts.length === 0 ? (
          <div className="empty-state">
            <Bell size={32} />
            <div className="empty-title">暂无告警记录</div>
          </div>
        ) : (
          <div className="alert-list">
            {alerts.map((alert, i) => (
              <div key={i} className="alert-item">
                <span className={`badge badge-${alert.alert_level || 'neutral'}`}>
                  {LEVEL_LABELS[alert.alert_level] || alert.alert_level}
                </span>
                <span className="alert-message">{alert.message}</span>
                <span className="alert-time mono">
                  {new Date(alert.timestamp * 1000).toLocaleString('zh-CN')}
                </span>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* ── 页脚 ── */}
      <footer className="app-footer">
        <span>跌倒风险预测系统</span>
        <span>多模态 AI 监测</span>
      </footer>
    </div>
  )
}
