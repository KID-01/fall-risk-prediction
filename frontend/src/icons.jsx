// 内联 SVG 图标集（Lucide 风格，保持零第三方依赖）
// 用法: <Icon name="shield" size={16} /> 或 <Shield size={16} />

const PATHS = {
  shield: <path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z" />,
  'audio-lines': <><path d="M2 10v3" /><path d="M6 6v11" /><path d="M10 3v18" /><path d="M14 8v7" /><path d="M18 5v13" /><path d="M22 10v3" /></>,
  activity: <path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2" />,
  moon: <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" />,
  sun: <><circle cx="12" cy="12" r="4" /><path d="M12 2v2" /><path d="M12 20v2" /><path d="m4.93 4.93 1.41 1.41" /><path d="m17.66 17.66 1.41 1.41" /><path d="M2 12h2" /><path d="M20 12h2" /><path d="m6.34 17.66-1.41 1.41" /><path d="m19.07 4.93-1.41 1.41" /></>,
  'refresh-cw': <><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" /><path d="M21 3v5h-5" /><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" /><path d="M8 16H3v5" /></>,
  play: <polygon points="6 3 20 12 6 21 6 3" />,
  square: <rect width="18" height="18" x="3" y="3" rx="2" />,
  'rotate-ccw': <><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" /><path d="M3 3v5h5" /></>,
  'alert-triangle': <><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" /><path d="M12 9v4" /><path d="M12 17h.01" /></>,
  'monitor-play': <><path d="M10 7.75a.75.75 0 0 1 1.142-.638l3.664 2.249a.75.75 0 0 1 0 1.278l-3.664 2.25a.75.75 0 0 1-1.142-.64z" /><path d="M12 17v4" /><path d="M8 21h8" /><rect x="2" y="3" width="20" height="14" rx="2" /></>,
  video: <><path d="m16 13 5.223 3.482a.5.5 0 0 0 .777-.416V7.87a.5.5 0 0 0-.752-.432L16 10.5" /><rect x="2" y="6" width="14" height="12" rx="2" /></>,
  bell: <><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" /><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" /></>,
  'chart-line': <><path d="M3 3v16a2 2 0 0 0 2 2h16" /><path d="m19 9-5 5-4-4-3 3" /></>,
  upload: <><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" x2="12" y1="3" y2="15" /></>,
  mic: <><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" /><path d="M19 10v2a7 7 0 0 1-14 0v-2" /><line x1="12" x2="12" y1="19" y2="22" /></>,
  'volume-x': <><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" /><line x1="22" x2="16" y1="9" y2="15" /><line x1="16" x2="22" y1="9" y2="15" /></>,
  check: <path d="M20 6 9 17l-5-5" />,
  'arrow-left': <><path d="m12 19-7-7 7-7" /><path d="M19 12H5" /></>,
  circle: <circle cx="12" cy="12" r="10" />,
  loader: <><path d="M21 12a9 9 0 1 1-6.219-8.56" /></>,
  clock: <><circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" /></>,
}

export default function Icon({ name, size = 16, className, style }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      style={style}
      aria-hidden="true"
    >
      {PATHS[name] || PATHS.circle}
    </svg>
  )
}

export const Shield = p => <Icon name="shield" {...p} />
export const AudioLines = p => <Icon name="audio-lines" {...p} />
export const Activity = p => <Icon name="activity" {...p} />
export const Moon = p => <Icon name="moon" {...p} />
export const Sun = p => <Icon name="sun" {...p} />
export const RefreshCw = p => <Icon name="refresh-cw" {...p} />
export const Play = p => <Icon name="play" {...p} />
export const Square = p => <Icon name="square" {...p} />
export const RotateCcw = p => <Icon name="rotate-ccw" {...p} />
export const AlertTriangle = p => <Icon name="alert-triangle" {...p} />
export const MonitorPlay = p => <Icon name="monitor-play" {...p} />
export const Video = p => <Icon name="video" {...p} />
export const Bell = p => <Icon name="bell" {...p} />
export const ChartLine = p => <Icon name="chart-line" {...p} />
export const Upload = p => <Icon name="upload" {...p} />
export const Mic = p => <Icon name="mic" {...p} />
export const VolumeX = p => <Icon name="volume-x" {...p} />
export const Check = p => <Icon name="check" {...p} />
export const ArrowLeft = p => <Icon name="arrow-left" {...p} />
export const Loader = p => <Icon name="loader" {...p} />
export const Clock = p => <Icon name="clock" {...p} />
