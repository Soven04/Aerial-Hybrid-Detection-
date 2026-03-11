import './ResultPanel.css'

export default function ResultPanel({ result, loading, error, preview }) {
  return (
    <div className="result-panel panel">
      <div className="panel-label">
        ◈ OUTPUT FEED
        {result && (
          <span className="result-meta mono">
            {result.count} OBJ · {result.inference_ms}ms · {result.image_w}×{result.image_h}
          </span>
        )}
      </div>

      <div className="result-viewport">
        {loading && (
          <div className="state-overlay">
            <div className="radar-ring" />
            <p className="mono state-text">SCANNING TARGET AREA</p>
            <p className="mono state-sub blink">■ INFERENCE RUNNING</p>
          </div>
        )}

        {error && !loading && (
          <div className="state-overlay error">
            <div className="error-icon">⚠</div>
            <p className="mono state-text">DETECTION FAILED</p>
            <p className="mono state-sub">{error}</p>
          </div>
        )}

        {!loading && !error && result && (
          <img src={result.image_b64} alt="detection result" className="result-img" />
        )}

        {!loading && !error && !result && preview && (
          <div className="state-overlay idle">
            <p className="mono state-text">AWAITING SCAN</p>
            <p className="mono state-sub blink">▸ PRESS RUN DETECTION</p>
          </div>
        )}

        {!loading && !error && !result && !preview && (
          <div className="state-overlay empty">
            <div className="grid-graphic">
              {Array.from({length:16}).map((_,i)=>(
                <div key={i} className="grid-cell" />
              ))}
            </div>
            <p className="mono state-text">NO FEED</p>
          </div>
        )}
      </div>

      {/* Stats bar */}
      {result && (
        <div className="stats-bar">
          <StatChip label="OBJECTS" value={result.count} accent />
          <StatChip label="TIME" value={`${result.inference_ms}ms`} />
          <StatChip label="DEVICE" value={result.device.toUpperCase()} />
          <StatChip label="SIZE" value={`${result.image_w}×${result.image_h}`} />
        </div>
      )}
    </div>
  )
}

function StatChip({ label, value, accent }) {
  return (
    <div className={`stat-chip ${accent ? 'accent' : ''}`}>
      <span className="chip-label mono">{label}</span>
      <span className="chip-val mono">{value}</span>
    </div>
  )
}
