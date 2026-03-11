import { useState } from 'react'
import './DetectionList.css'

export default function DetectionList({ detections }) {
  const [filter, setFilter] = useState('')

  if (!detections || detections.length === 0) return null

  const filtered = detections.filter(d =>
    d.label.toLowerCase().includes(filter.toLowerCase())
  )

  // Group by label for summary
  const counts = {}
  detections.forEach(d => { counts[d.label] = (counts[d.label] || 0) + 1 })

  return (
    <div className="det-list panel">
      <div className="panel-label">
        ◈ DETECTIONS
        <span className="mono" style={{ color: 'var(--text-muted)', fontSize: '0.6rem' }}>
          {detections.length} TOTAL
        </span>
      </div>

      {/* Class summary */}
      <div className="class-summary">
        {Object.entries(counts).map(([cls, cnt]) => {
          const d = detections.find(x => x.label === cls)
          return (
            <div key={cls} className="class-tag" style={{ '--clr': d?.color }}>
              <span className="tag-dot" />
              <span className="mono tag-name">{cls}</span>
              <span className="mono tag-count">{cnt}</span>
            </div>
          )
        })}
      </div>

      {/* Filter */}
      <input
        className="det-filter mono"
        placeholder="FILTER CLASS..."
        value={filter}
        onChange={e => setFilter(e.target.value)}
      />

      {/* List */}
      <div className="det-rows">
        {filtered.map((d, i) => (
          <div key={i} className="det-row" style={{ '--clr': d.color }}>
            <div className="det-idx mono">{String(i + 1).padStart(2, '0')}</div>
            <div className="det-color-bar" />
            <div className="det-info">
              <span className="det-label">{d.label}</span>
              <span className="det-coords mono">
                [{d.box.map(v => Math.round(v)).join(', ')}]
              </span>
            </div>
            <div className="det-score mono">{(d.score * 100).toFixed(1)}%</div>
            <div className="score-bar-wrap">
              <div className="score-bar" style={{ width: `${d.score * 100}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
