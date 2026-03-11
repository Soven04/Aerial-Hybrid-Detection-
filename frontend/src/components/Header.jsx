import './Header.css'

export default function Header({ health }) {
  const online = health?.status === 'ok'
  const device = health?.device || '---'

  return (
    <header className="header">
      <div className="header-left">
        <div className="logo-mark">
          <svg width="38" height="38" viewBox="0 0 38 38">
            <polygon points="19,2 36,10 36,28 19,36 2,28 2,10"
              fill="none" stroke="#00d4ff" strokeWidth="1.5" />
            <polygon points="19,9 29,14 29,24 19,29 9,24 9,14"
              fill="none" stroke="#00d4ff" strokeWidth="0.8" opacity="0.5" />
            <circle cx="19" cy="19" r="3" fill="#00d4ff" />
            <line x1="19" y1="2"  x2="19" y2="9"  stroke="#00d4ff" strokeWidth="1"/>
            <line x1="19" y1="29" x2="19" y2="36" stroke="#00d4ff" strokeWidth="1"/>
            <line x1="2"  y1="10" x2="9"  y2="14" stroke="#00d4ff" strokeWidth="1"/>
            <line x1="29" y1="24" x2="36" y2="28" stroke="#00d4ff" strokeWidth="1"/>
          </svg>
        </div>
        <div className="header-titles">
          <h1 className="title-main">AERIALDET</h1>
          <span className="title-sub">HYBRID · SPARSE-RCNN + TRANSFORMER</span>
        </div>
      </div>

      <div className="header-right">
        <div className="sys-stat">
          <span className="stat-label mono">SYS</span>
          <span className={`status-dot ${online ? 'online' : 'offline'}`} />
          <span className="stat-val mono">{online ? 'ONLINE' : 'OFFLINE'}</span>
        </div>
        <div className="sys-stat">
          <span className="stat-label mono">DEV</span>
          <span className="stat-val mono accent">{device.toUpperCase()}</span>
        </div>
        <div className="sys-stat">
          <span className="stat-label mono">VER</span>
          <span className="stat-val mono">1.0.0</span>
        </div>
      </div>
    </header>
  )
}
