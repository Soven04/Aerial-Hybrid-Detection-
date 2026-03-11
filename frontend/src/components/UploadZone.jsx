import { useRef, useState } from 'react'
import './UploadZone.css'

export default function UploadZone({
  preview, onFile, onDetect, loading, hasFile,
  scoreThresh, nmsThresh, onScoreChange, onNmsChange
}) {
  const inputRef  = useRef()
  const [drag, setDrag] = useState(false)

  const handle = (f) => {
    if (f && f.type.startsWith('image/')) onFile(f)
  }

  return (
    <div className="upload-panel panel">
      <div className="panel-label">◈ INPUT FEED</div>

      {/* Drop zone */}
      <div
        className={`drop-zone ${drag ? 'drag' : ''} ${hasFile ? 'has-file' : ''}`}
        onClick={() => inputRef.current.click()}
        onDragOver={e => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={e => { e.preventDefault(); setDrag(false); handle(e.dataTransfer.files[0]) }}
      >
        <input
          ref={inputRef} type="file" accept="image/*" style={{ display: 'none' }}
          onChange={e => handle(e.target.files[0])}
        />

        {preview ? (
          <div className="preview-wrap">
            <img src={preview} alt="input" className="preview-img" />
            <div className="preview-overlay">
              <span className="mono">CLICK TO CHANGE</span>
            </div>
            {/* Targeting corners */}
            <div className="corner tl" /><div className="corner tr" />
            <div className="corner bl" /><div className="corner br" />
          </div>
        ) : (
          <div className="drop-hint">
            <svg className="drop-icon" viewBox="0 0 64 64" width="56" height="56">
              <rect x="8" y="8" width="48" height="48" rx="4"
                fill="none" stroke="currentColor" strokeWidth="1.5" strokeDasharray="4 3" />
              <line x1="32" y1="20" x2="32" y2="44" stroke="currentColor" strokeWidth="2"/>
              <line x1="20" y1="32" x2="44" y2="32" stroke="currentColor" strokeWidth="2"/>
              <circle cx="32" cy="32" r="8" fill="none" stroke="currentColor" strokeWidth="1.2"/>
            </svg>
            <p className="drop-text">DROP AERIAL IMAGE</p>
            <p className="drop-sub mono">jpg · png · tif · webp</p>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="controls-block">
        <div className="control-row">
          <label className="ctrl-label mono">SCORE THR</label>
          <input
            type="range" min="0.1" max="0.9" step="0.05"
            value={scoreThresh}
            onChange={e => onScoreChange(parseFloat(e.target.value))}
            className="slider"
          />
          <span className="ctrl-val mono">{scoreThresh.toFixed(2)}</span>
        </div>
        <div className="control-row">
          <label className="ctrl-label mono">NMS THR</label>
          <input
            type="range" min="0.1" max="0.9" step="0.05"
            value={nmsThresh}
            onChange={e => onNmsChange(parseFloat(e.target.value))}
            className="slider"
          />
          <span className="ctrl-val mono">{nmsThresh.toFixed(2)}</span>
        </div>
      </div>

      {/* Detect button */}
      <button
        className={`detect-btn ${loading ? 'loading' : ''}`}
        onClick={onDetect}
        disabled={!hasFile || loading}
      >
        {loading ? (
          <>
            <span className="spin-ring" />
            <span>PROCESSING</span>
          </>
        ) : (
          <>
            <span className="btn-icon">⬡</span>
            <span>RUN DETECTION</span>
          </>
        )}
      </button>
    </div>
  )
}
