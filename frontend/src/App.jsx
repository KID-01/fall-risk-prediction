import React, { useState, useEffect, useRef, useCallback } from 'react'
import * as echarts from 'echarts'
import AudioMonitor from './AudioMonitor'
import { Shield, Moon, Sun, RefreshCw, Play, Square, RotateCcw, AlertTriangle, MonitorPlay, Video, Bell, ChartLine, AudioLines, Loader } from './icons'

const API_BASE = '/api/v1'
const LEVEL_LABELS = { low: '低风险', attention: '关注级', warning: '预警级', critical: '高危级' }

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

function EzvizPlayer({ active, config, setPlayerState, setPlayerError }) {
  const hostRef = useRef(null)

  useEffect(() => {
    if (!active || !config || !hostRef.current) return undefined

    let cancelled = false
    let player
    setPlayerState('loading')
    setPlayerError('')

    import('ezuikit-js')
      .then(({ EZUIKitPlayer }) => {
        if (cancelled || !hostRef.current) return
        const width = Math.max(320, Math.floor(hostRef.current.clientWidth))
        player = new EZUIKitPlayer({
          id: 'ezviz-player',
          accessToken: config.accessToken,
          url: config.url,
          width,
          height: Math.floor(width * 9 / 16),
          template: 'pcLive',
          audio: false,
          handleSuccess: () => {
            if (!cancelled) setPlayerState('playing')
          },
          handleError: error => {
            if (cancelled) return
            console.error('EZUIKit 播放失败', error)
            const encrypted = error?.type === 'handleRunTimeInfoError' && error?.data?.nErrorCode === 5
            setPlayerState('error')
            setPlayerError(encrypted
              ? '设备已启用视频加密，需要设备验证码。'
              : '萤石视频播放失败，请确认设备在线并刷新播放授权。')
          },
        })
      })
      .catch(error => {
        if (cancelled) return
        console.error('EZUIKit 初始化失败', error)
        setPlayerState('error')
        setPlayerError('播放器初始化失败，请检查浏览器兼容性和萤石播放权限。')
      })

    return () => {
      cancelled = true
      try {
        player?.stop?.()
        player?.destroy?.()
      } catch (_) { /* 播放器释放失败不影响后端监控 */ }
      setPlayerState('idle')
    }
  }, [active, config, setPlayerError, setPlayerState])

  return <div ref={hostRef} className="ezviz-player-host"><div id="ezviz-player" /></div>
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
  const [playerConfig, setPlayerConfig] = useState(null)
  const [playerState, setPlayerState] = useState('idle')
  const [playerError, setPlayerError] = useState('')
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('theme')
    return saved || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
  })
  const wsRef = useRef(null)
  const videoWsRef = useRef(null)
  const videoImgRef = useRef(null)
  const gaugeRef = useRef(null)
  const trendRef = useRef(null)

  const selectedDevice = devices.find(device => device.device_id === selectedDeviceId)

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
      if (msg.type === 'alert') { fetchAlerts(); fetchStatus() }
    }

    return () => ws.close()
  }, [])

  // ── 视频 WebSocket ──
  useEffect(() => {
    if (videoTab !== 'analysis') return undefined

    let stopped = false
    let currentObjectUrl = ''
    const connectVideo = () => {
      if (stopped) return
      const wsUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/video`
      const ws = new WebSocket(wsUrl)
      videoWsRef.current = ws

      ws.onmessage = (event) => {
        if (videoImgRef.current && event.data instanceof Blob) {
          const url = URL.createObjectURL(event.data)
          if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl)
          currentObjectUrl = url
          videoImgRef.current.src = url
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
    const score = status.last_feature ? 50 : 0
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
          formatter: '{value}',
          fontSize: 36,
          fontWeight: 700,
          color: textColor,
          offsetCenter: [0, '55%'],
        },
        data: [{ value: score, name: '风险评分' }],
      }],
    })
    return () => chart.dispose()
  }, [status, theme])

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
    const scores = riskHistory.map(r => r.risk_score || 0).reverse()
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
        type: 'value', min: 0, max: 100,
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
      }],
    })
    return () => chart.dispose()
  }, [riskHistory, theme])

  // ── 控制操作 ──
  const startMonitor = async () => {
    localStorage.setItem('monitor_person_id', personId)
    localStorage.setItem('monitor_audio_source', audioSource)
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
          body: JSON.stringify({ source, person_id: personId, audio_source: audioSource }),
        })
      }

      if (!response.ok) {
        setControlError(await readError(response))
        return
      }
      if (sourceMode === 'ezviz') setPlayerConfig(await response.json())
      else setPlayerConfig(null)
      setVideoTab('analysis')
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
      setPlayerError('')
      setPlayerState('idle')
      setVideoTab('analysis')
      fetchStatus()
    } catch (_) {
      setControlError('无法连接后端服务，请确认 FastAPI 已启动')
    }
  }
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
              <option value="auto">音频: 自动(跟随视频源)</option>
              <option value="off">音频: 关闭</option>
              <option value="mic">音频: 麦克风</option>
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
              <button type="button" role="tab" aria-selected={videoTab === 'raw'} className={videoTab === 'raw' ? 'active' : ''} onClick={() => setVideoTab('raw')} disabled={sourceMode !== 'ezviz'}>萤石原始画面</button>
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
        <div className="video-frame-wrap">
          {videoTab === 'analysis' ? (
            <>
              <img ref={videoImgRef} alt="AI 分析实时画面" className="video-frame" />
              {!status.is_running && (
                <div className="video-placeholder">
                  <MonitorPlay size={30} />
                  <span>点击「启动监控」开始查看分析画面</span>
                </div>
              )}
            </>
          ) : videoTab === 'raw' ? (
            <>
              {playerConfig
                ? <EzvizPlayer active={videoTab === 'raw'} config={playerConfig} setPlayerState={setPlayerState} setPlayerError={setPlayerError} />
                : <div className="video-placeholder"><Video size={30} /><span>选择在线设备并启动监控后显示原始画面</span></div>}
              {playerState === 'error' && (
                <div className="video-error" role="alert"><AlertTriangle size={14} /> {playerError}</div>
              )}
            </>
          ) : (
            <AudioMonitor
              pipelineAudio={{
                enabled: status.audio_enabled,
                source: status.audio_source,
                chunksProcessed: status.audio_chunks_processed,
                error: status.audio_error,
                lastResult: status.last_audio_result,
                lastAlert: status.last_alert,
              }}
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
            {status.last_alert ? status.last_alert.message : '系统运行正常，持续监测中'}
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
