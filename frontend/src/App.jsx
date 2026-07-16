import React, { useState, useEffect, useRef, useCallback } from 'react'
import * as echarts from 'echarts'

const API_BASE = '/api/v1'
const LEVEL_LABELS = { low: '低风险', attention: '关注级', warning: '预警级', critical: '高危级' }
const LEVEL_COLORS = { low: '#52c41a', attention: '#faad14', warning: '#fa8c16', critical: '#f5222d' }

export default function App() {
  const [status, setStatus] = useState({
    is_running: false,
    current_risk_level: 'low',
    current_risk_label: '低风险',
    baseline_ready: false,
    baseline_samples: 0,
    last_feature: null,
    last_alert: null,
  })
  const [alerts, setAlerts] = useState([])
  const [riskHistory, setRiskHistory] = useState([])
  const [stats, setStats] = useState({})
  const [connected, setConnected] = useState(false)
  const wsRef = useRef(null)
  const gaugeRef = useRef(null)
  const trendRef = useRef(null)
  const radarRef = useRef(null)

  // ── 获取数据 ──
  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/risk/current`)
      const data = await res.json()
      setStatus(data)
    } catch (e) { console.error('状态获取失败:', e) }
  }, [])

  const fetchAlerts = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/alerts?limit=10`)
      const data = await res.json()
      setAlerts(data.alerts || [])
    } catch (e) { console.error('告警获取失败:', e) }
  }, [])

  const fetchRiskHistory = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/risk/history?hours=24&limit=100`)
      const data = await res.json()
      setRiskHistory(data.records || [])
    } catch (e) { console.error('历史获取失败:', e) }
  }, [])

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/stats?hours=24`)
      const data = await res.json()
      setStats(data)
    } catch (e) { console.error('统计获取失败:', e) }
  }, [])

  // ── WebSocket 实时推送 ──
  useEffect(() => {
    const wsUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/alerts`
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => { setConnected(true); console.log('WebSocket已连接') }
    ws.onclose = () => { setConnected(false); console.log('WebSocket已断开') }
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data)
      if (msg.type === 'alert') {
        fetchAlerts()
        fetchStatus()
      }
    }
    ws.onerror = (e) => console.error('WebSocket错误:', e)

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

  // ── 风险评分仪表盘 ──
  useEffect(() => {
    if (!gaugeRef.current) return
    const chart = echarts.init(gaugeRef.current)
    const score = status.last_feature ? 50 : 0
    const level = status.current_risk_level || 'low'
    chart.setOption({
      series: [{
        type: 'gauge',
        min: 0, max: 100,
        axisLine: { lineStyle: { width: 20, color: [
          [0.3, '#52c41a'], [0.5, '#faad14'], [0.75, '#fa8c16'], [1, '#f5222d']
        ]}},
        pointer: { width: 5 },
        detail: { formatter: '{value}', fontSize: 32 },
        data: [{ value: score, name: '风险评分' }]
      }]
    })
    return () => chart.dispose()
  }, [status])

  // ── 风险趋势折线图 ──
  useEffect(() => {
    if (!trendRef.current || riskHistory.length === 0) return
    const chart = echarts.init(trendRef.current)
    const times = riskHistory.map(r => new Date(r.timestamp * 1000).toLocaleTimeString()).reverse()
    const scores = riskHistory.map(r => r.risk_score || 0).reverse()
    chart.setOption({
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: times, axisLabel: { fontSize: 12 } },
      yAxis: { type: 'value', min: 0, max: 100 },
      series: [{
        type: 'line', data: scores, smooth: true,
        lineStyle: { width: 3, color: '#1890ff' },
        areaStyle: { color: 'rgba(24,144,255,0.1)' },
      }]
    })
    return () => chart.dispose()
  }, [riskHistory])

  // ── 控制操作 ──
  const startMonitor = async () => {
    await fetch(`${API_BASE}/stream/start`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: '0', person_id: 'default' })
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

  return (
    <div className="dashboard">
      {/* 头部 */}
      <div className="dashboard-header">
        <h1>跌倒风险预测系统</h1>
        <div>
          <span className={`status-dot ${connected ? 'online' : 'offline'}`} />
          {connected ? '实时连接' : '未连接'}
        </div>
      </div>

      {/* 控制按钮 */}
      <div className="controls">
        <button className="btn btn-primary" onClick={startMonitor} disabled={status.is_running}>
          启动监控
        </button>
        <button className="btn btn-danger" onClick={stopMonitor} disabled={!status.is_running}>
          停止监控
        </button>
        <button className="btn" onClick={resetBaseline} style={{ background: '#d9d9d9' }}>
          重置基线
        </button>
      </div>

      {/* 风险等级大卡片 */}
      <div className={`risk-card ${level}`}>
        <div className="level-label" style={{ color: levelColor }}>{levelLabel}</div>
        <div className="risk-message">
          {status.last_alert ? status.last_alert.message : '系统运行正常，持续监测中'}
        </div>
        <div style={{ marginTop: 16, color: 'var(--muted)' }}>
          基线状态: {status.baseline_ready ? `已就绪 (${status.baseline_samples}样本)` : `采集中 (${status.baseline_samples}/100)`}
          {' | '}处理帧数: {status.frames_processed || 0}
          {' | '}有效帧: {status.frames_valid || 0}
        </div>
      </div>

      {/* 图表网格 */}
      <div className="chart-grid">
        <div className="chart-card">
          <h3>当前风险评分</h3>
          <div ref={gaugeRef} className="chart-container" />
        </div>
        <div className="chart-card">
          <h3>近24小时风险趋势</h3>
          <div ref={trendRef} className="chart-container" />
        </div>
      </div>

      {/* 告警列表 */}
      <div className="alert-list">
        <h3>最新告警</h3>
        {alerts.length === 0 ? (
          <div style={{ color: 'var(--muted)', padding: '20px 0' }}>暂无告警记录</div>
        ) : (
          alerts.map((alert, i) => (
            <div key={i} className="alert-item">
              <span className={`alert-badge ${alert.alert_level}`}>
                {LEVEL_LABELS[alert.alert_level] || alert.alert_level}
              </span>
              <span style={{ flex: 1 }}>{alert.message}</span>
              <span style={{ color: 'var(--muted)', fontSize: 14 }}>
                {new Date(alert.timestamp * 1000).toLocaleString()}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
