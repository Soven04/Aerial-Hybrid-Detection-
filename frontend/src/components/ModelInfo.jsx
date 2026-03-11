import './ModelInfo.css'

const ARCH_BLOCKS = [
  { id: 'backbone', label: 'BACKBONE',     detail: 'ResNet-style C3/C4/C5', color: '#4363d8' },
  { id: 'fpn',      label: 'FPN NECK',     detail: 'P3–P6 · 256 channels',  color: '#911eb4' },
  { id: 'enc',      label: 'TRANSFORMER',  detail: 'Deformable Encoder ×6', color: '#00d4ff' },
  { id: 'prop',     label: 'PROPOSALS',    detail: '100 Learnable Queries',  color: '#ff6b35' },
  { id: 'rcnn',     label: 'SPARSE RCNN',  detail: 'Dynamic Head ×6 stages', color: '#22ff88' },
  { id: 'out',      label: 'OUTPUT',       detail: 'cls logits + boxes',     color: '#ffe119' },
]

export default function ModelInfo({ info }) {
  return (
    <div className="model-page">
      <div className="model-grid">
        {/* Architecture diagram */}
        <div className="panel arch-panel">
          <div className="panel-label">◈ ARCHITECTURE PIPELINE</div>
          <div className="arch-flow">
            {ARCH_BLOCKS.map((b, i) => (
              <div key={b.id} className="arch-block" style={{ '--bclr': b.color }}>
                <div className="block-box">
                  <span className="block-label mono">{b.label}</span>
                  <span className="block-detail">{b.detail}</span>
                </div>
                {i < ARCH_BLOCKS.length - 1 && (
                  <div className="arch-arrow">▼</div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Stats */}
        <div className="panel stats-panel">
          <div className="panel-label">◈ MODEL PARAMETERS</div>
          {info ? (
            <div className="info-rows">
              {[
                ['Architecture', info.architecture],
                ['Total Params',     `${info.total_params_M}M`],
                ['Trainable Params', `${info.trainable_params_M}M`],
                ['Classes',          info.num_classes],
                ['Proposals',        info.num_proposals],
                ['RCNN Stages',      info.num_stages],
                ['Encoder Layers',   info.encoder_layers],
                ['Feature Dim',      info.d_model],
                ['Device',           info.device?.toUpperCase()],
              ].map(([k, v]) => (
                <div key={k} className="info-row">
                  <span className="info-key mono">{k}</span>
                  <span className="info-val">{v}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="mono" style={{color:'var(--text-muted)',fontSize:'0.7rem',marginTop:'1rem'}}>
              LOADING MODEL INFO...
            </p>
          )}
        </div>

        {/* Class list */}
        <div className="panel classes-panel">
          <div className="panel-label">◈ DOTA v1.0 CLASSES (15)</div>
          <div className="class-grid">
            {[
              'plane','baseball-diamond','bridge','ground-track-field',
              'small-vehicle','large-vehicle','ship','tennis-court',
              'basketball-court','storage-tank','soccer-ball-field',
              'roundabout','harbor','swimming-pool','helicopter'
            ].map((cls, i) => {
              const COLORS = [
                '#e6194b','#3cb44b','#ffe119','#4363d8','#f58231',
                '#911eb4','#42d4f4','#f032e6','#bfef45','#fabed4',
                '#469990','#dcbeff','#9A6324','#fffac8','#800000',
              ]
              return (
                <div key={cls} className="cls-item" style={{ '--clr': COLORS[i] }}>
                  <span className="cls-id mono">{String(i).padStart(2,'0')}</span>
                  <span className="cls-dot" />
                  <span className="cls-name">{cls}</span>
                </div>
              )
            })}
          </div>
        </div>

        {/* Key innovations */}
        <div className="panel innov-panel">
          <div className="panel-label">◈ KEY INNOVATIONS</div>
          <div className="innov-list">
            {[
              { icon: '⬡', title: 'Sparse Proposals',       desc: '100 learnable queries replace 100K+ dense anchors — eliminates redundant computation' },
              { icon: '◈', title: 'Dynamic Conv',           desc: 'Instance-conditioned kernels: each proposal uses its own generated conv weights' },
              { icon: '⟳', title: 'Iterative Refinement',  desc: '6 cascaded stages progressively sharpen box regression and classification' },
              { icon: '∞', title: 'Deformable Attention',  desc: 'Offsets learned per head capture irregular spatial patterns in aerial scenes' },
              { icon: '⊞', title: 'Multi-Scale FPN',       desc: 'P3–P6 feature pyramid handles extreme scale variation from 1m to 50m objects' },
            ].map(item => (
              <div key={item.title} className="innov-item">
                <span className="innov-icon">{item.icon}</span>
                <div>
                  <p className="innov-title">{item.title}</p>
                  <p className="innov-desc">{item.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
