import React, { useState, useEffect, useRef, useCallback } from 'react'
import * as echarts from 'echarts'

const API_BASE = '/api/v1'
const LEVEL_LABELS = { low: '低风险', attention: '关注级', warning: '预警级', critical: '高危级' }
const LEVEL_COLORS = { low: '#22c55e', attention: '#eab308', warning: '#f97316', critical: '#ef4444' }
const LEVEL_ICONS = { low: '✓', attention: '◉', warning: '⚠', critical: '✕' }

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
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('theme')
    return saved || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
  })
  const wsRef = useRef(null)
  const gaugeRef = useRef(null)
  const trendRef = useRef(null)

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
    const isDark = theme === 'dark'
    const textColor = isDark ? '#e2e8f0' : '#1e293b'
    const labelColor = isDark ? '#cbd5e1' : '#64748b'
    const tickColor = isDark ? '#94a3b8' : '#94a3b8'
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
              [0.3, '#22c55e'],
              [0.5, '#eab308'],
              [0.75, '#f97316'],
              [1, '#ef4444'],
            ],
          },
        },
        pointer: { width: 5, itemStyle: { color: '#3b82f6' } },
        axisTick: { distance: -18, length: 6, lineStyle: { width: 1, color: tickColor } },
        splitLine: { distance: -22, length: 14, lineStyle: { width: 2, color: tickColor } },
        axisLabel: { distance: 36, fontSize: 13, fontWeight: 600, color: labelColor },
        anchor: { show: true, size: 14, itemStyle: { color: '#3b82f6' } },
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
    const isDark = theme === 'dark'
    const textColor = isDark ? '#e2e8f0' : '#1e293b'
    const mutedColor = isDark ? '#64748b' : '#94a3b8'
    const axisColor = isDark ? '#475569' : '#e2e8f0'
    const splitColor = isDark ? '#334155' : '#f1f5f9'
    const tooltipBg = isDark ? '#1e293b' : '#ffffff'
    const tooltipBorder = isDark ? '#475569' : '#e2e8f0'
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
        lineStyle: { width: 3, color: '#3b82f6' },
        itemStyle: { color: '#3b82f6' },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(59,130,246,0.15)' },
            { offset: 1, color: 'rgba(59,130,246,0.01)' },
          ]),
        },
      }],
    })
    return () => chart.dispose()
  }, [riskHistory, theme])

  // ── 控制操作 ──
  const startMonitor = async () => {
    await fetch(`${API_BASE}/stream/start`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: '0', person_id: 'default' }),
    })
    fetchStatus()
  }
  const stopMonitor = async () => {
    await fetch(`${API_BASE}/stream/stop`, { method: 'POST' })
    fetchStatus()
  }
  const resetBaseline = async () => {
    await fetch(`${API_BASE}/baseline/reset`, { method: 'POST' })
    fetchStatus()
  }

  const level = status.current_risk_level || 'low'
  const levelLabel = LEVEL_LABELS[level] || '低风险'
  const levelColor = LEVEL_COLORS[level] || LEVEL_COLORS.low
  const levelIcon = LEVEL_ICONS[level] || '✓'

  return (
    <div className="dashboard">
      {/* ── 顶部标题栏 ── */}
      <div className="dashboard-header">
        <div className="header-left">
          <h1>跌倒风险预测系统</h1>
          <div className="subtitle">基于多模态 AI 监测 · 家属端实时看板</div>
        </div>
        <div className="header-right">
          <button className="theme-toggle" onClick={toggleTheme} title={theme === 'light' ? '切换暗色模式' : '切换亮色模式'}>
            {theme === 'light' ? '🌙' : '☀️'}
          </button>
          <span className={`connection-badge ${connected ? 'online' : 'offline'}`}>
            <span className="dot" />
            {connected ? '实时连接' : '未连接'}
          </span>
        </div>
      </div>

      {/* ── 控制按钮 ── */}
      <div className="controls">
        <button className="btn btn-primary" onClick={startMonitor} disabled={status.is_running}>
          ▶ 启动监控
        </button>
        <button className="btn btn-danger" onClick={stopMonitor} disabled={!status.is_running}>
          ■ 停止监控
        </button>
        <button className="btn btn-secondary" onClick={resetBaseline}>
          ↻ 重置基线
        </button>
      </div>

      {/* ── 风险等级大卡片 ── */}
      <div className={`risk-card ${level}`}>
        <div className="level-label" style={{ color: levelColor }}>{levelLabel}</div>
        <div className="risk-message">
          {status.last_alert ? status.last_alert.message : '系统运行正常，持续监测中'}
        </div>
        <div className="meta-row">
          <div className="meta-item">
            <div className="meta-value">{status.baseline_ready ? '✓' : `${status.baseline_samples || 0}/100`}</div>
            <div className="meta-label">基线采集</div>
          </div>
          <div className="meta-item">
            <div className="meta-value">{status.frames_processed || 0}</div>
            <div className="meta-label">处理帧数</div>
          </div>
          <div className="meta-item">
            <div className="meta-value">{status.frames_valid || 0}</div>
            <div className="meta-label">有效帧数</div>
          </div>
        </div>
      </div>

      {/* ── 图表网格 ── */}
      <div className="chart-grid">
        <div className="chart-card">
          <h3>当前风险评分</h3>
          <div ref={gaugeRef} className="chart-container" />
        </div>
        <div className="chart-card">
          <h3>近24小时风险趋势</h3>
          {riskHistory.length === 0 ? (
            <div className="chart-empty">
              <span className="empty-icon">📊</span>
              <span>暂无历史数据</span>
              <span style={{ fontSize: 13 }}>启动监控后数据将在此展示</span>
            </div>
          ) : (
            <div ref={trendRef} className="chart-container" />
          )}
        </div>
      </div>

      {/* ── 告警列表 ── */}
      <div className="alert-section">
        <h3>最新告警</h3>
        {alerts.length === 0 ? (
          <div className="alert-empty">
            <span className="empty-icon">🔔</span>
            <span>暂无告警记录</span>
          </div>
        ) : (
          alerts.map((alert, i) => (
            <div key={i} className="alert-item">
              <span className={`alert-badge ${alert.alert_level}`}>
                {LEVEL_LABELS[alert.alert_level] || alert.alert_level}
              </span>
              <span className="alert-message">{alert.message}</span>
              <span className="alert-time">
                {new Date(alert.timestamp * 1000).toLocaleString('zh-CN')}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}