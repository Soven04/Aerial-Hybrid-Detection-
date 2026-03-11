import { useState, useEffect } from 'react'
import Header from './components/Header.jsx'
import UploadZone from './components/UploadZone.jsx'
import ResultPanel from './components/ResultPanel.jsx'
import DetectionList from './components/DetectionList.jsx'
import ModelInfo from './components/ModelInfo.jsx'
import StatusBar from './components/StatusBar.jsx'
import './App.css'

const API = 'http://127.0.0.1:8000'

export default function App() {
  const [file,        setFile]        = useState(null)
  const [preview,     setPreview]     = useState(null)
  const [result,      setResult]      = useState(null)
  const [loading,     setLoading]     = useState(false)
  const [error,       setError]       = useState(null)
  const [modelInfo,   setModelInfo]   = useState(null)
  const [health,      setHealth]      = useState(null)
  const [scoreThresh, setScoreThresh] = useState(0.35)
  const [nmsThresh,   setNmsThresh]   = useState(0.50)
  const [activeTab,   setActiveTab]   = useState('detect')

  // Fetch model info on mount
  useEffect(() => {
    fetch(`${API}/health`).then(r => r.json()).then(setHealth).catch(() => {})
    fetch(`${API}/model/info`).then(r => r.json()).then(setModelInfo).catch(() => {})
  }, [])

  const handleFile = (f) => {
    setFile(f)
    setResult(null)
    setError(null)
    setPreview(URL.createObjectURL(f))
  }

  const handleDetect = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)

    const form = new FormData()
    form.append('file', file)

    try {
      const res = await fetch(
        `${API}/detect?score_threshold=${scoreThresh}&nms_threshold=${nmsThresh}`,
        { method: 'POST', body: form }
      )
      if (!res.ok) throw new Error(`Server error ${res.status}`)
      const data = await res.json()
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <Header health={health} />

      <nav className="tab-bar">
        {['detect', 'model'].map(tab => (
          <button
            key={tab}
            className={`tab-btn ${activeTab === tab ? 'active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            <span className="tab-label">
              {tab === 'detect' ? '⬡ DETECTION' : '◈ MODEL INFO'}
            </span>
          </button>
        ))}
      </nav>

      <main className="main-content">
        {activeTab === 'detect' ? (
          <div className="detect-layout">
            {/* Left column */}
            <div className="left-col">
              <UploadZone
                preview={preview}
                onFile={handleFile}
                onDetect={handleDetect}
                loading={loading}
                hasFile={!!file}
                scoreThresh={scoreThresh}
                nmsThresh={nmsThresh}
                onScoreChange={setScoreThresh}
                onNmsChange={setNmsThresh}
              />
            </div>

            {/* Right column */}
            <div className="right-col">
              <ResultPanel
                result={result}
                loading={loading}
                error={error}
                preview={preview}
              />
              {result && (
                <DetectionList detections={result.detections} />
              )}
            </div>
          </div>
        ) : (
          <ModelInfo info={modelInfo} />
        )}
      </main>

      <StatusBar health={health} result={result} loading={loading} />
    </div>
  )
}
