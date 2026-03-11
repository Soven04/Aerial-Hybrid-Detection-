import { useState, useEffect } from 'react'
import './StatusBar.css'

export default function StatusBar({ health, result, loading }) {
  const [time, setTime] = useState(new Date())

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  const fmt = (d) =>
    `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}:${String(d.getSeconds()).padStart(2,'0')} UTC`

  return (
    <footer className="status-bar">
      <div className="sb-left mono">
        <span className={`sb-dot ${health?.status === 'ok' ? 'ok' : 'err'}`} />
        <span>API: {health?.status === 'ok' ? 'CONNECTED' : 'OFFLINE'}</span>
        {health?.device && <span className="sb-sep">·</span>}
        {health?.device && <span>DEV: {health.device.toUpperCase()}</span>}
      </div>
      <div className="sb-center mono">
        {loading ? (
          <span className="blink">■ INFERENCE IN PROGRESS</span>
        ) : result ? (
          <span>LAST SCAN: {result.count} OBJ · {result.inference_ms}ms</span>
        ) : (
          <span>AWAITING TARGET INPUT</span>
        )}
      </div>
      <div className="sb-right mono">
        <span>AERIALDET-HYBRID v1.0</span>
        <span className="sb-sep">·</span>
        <span>{fmt(time)}</span>
      </div>
    </footer>
  )
}
