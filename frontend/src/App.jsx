import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'

// API base URL - use direct backend URL for dev, /api proxy for production
const API_BASE = window.location.hostname === 'localhost' 
  ? 'http://localhost:8200'
  : '/api'

function App() {
  // State
  const [patients, setPatients] = useState([])
  const [selectedPatient, setSelectedPatient] = useState(null) // null = global mode
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [topK, setTopK] = useState(5)
  const [useReranker, setUseReranker] = useState(false)
  const [lastMetrics, setLastMetrics] = useState(null)
  const [lastChunks, setLastChunks] = useState([])
  const [lastToolCalls, setLastToolCalls] = useState([])
  const [expandedChunks, setExpandedChunks] = useState({})
  const [stats, setStats] = useState(null)
  const [showSettings, setShowSettings] = useState(false)
  const [showEvidence, setShowEvidence] = useState(false)
  
  // Navigation state (keyboard nav like cv_website)
  const [menuIdx, setMenuIdx] = useState(0)
  const menuItems = [
    { label: 'global search', action: () => setSelectedPatient(null) },
    { label: 'select patient', action: () => setShowSettings(s => !s) },
    { label: 'evidence', action: () => setShowEvidence(s => !s) },
  ]
  
  const inputRef = useRef(null)
  const messagesEndRef = useRef(null)

  // Load data on mount
  useEffect(() => {
    fetchPatients()
    fetchStats()
  }, [])

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Skip if typing in input
      if (document.activeElement === inputRef.current) return
      
      if (e.key === 'ArrowUp' || e.key === 'k') {
        e.preventDefault()
        setMenuIdx(i => (i - 1 + menuItems.length) % menuItems.length)
      } else if (e.key === 'ArrowDown' || e.key === 'j') {
        e.preventDefault()
        setMenuIdx(i => (i + 1) % menuItems.length)
      } else if (e.key === 'Enter' && !loading) {
        e.preventDefault()
        menuItems[menuIdx].action()
      } else if (e.key === '/' || e.key === 'i') {
        e.preventDefault()
        inputRef.current?.focus()
      } else if (e.key === 'Escape') {
        inputRef.current?.blur()
        setShowSettings(false)
        setShowEvidence(false)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [menuIdx, menuItems, loading])

  const fetchPatients = async () => {
    try {
      const res = await fetch(`${API_BASE}/patients?limit=100`)
      const data = await res.json()
      setPatients(data)
    } catch (err) {
      console.error('Failed to fetch patients:', err)
    }
  }

  const fetchStats = async () => {
    try {
      const res = await fetch(`${API_BASE}/stats`)
      const data = await res.json()
      setStats(data)
    } catch (err) {
      console.error('Failed to fetch stats:', err)
    }
  }

  const sendMessage = async () => {
    if (!input.trim() || loading) return

    const userMessage = input.trim()
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: userMessage }])
    setLoading(true)
    setLastChunks([])
    setLastToolCalls([])

    try {
      const body = {
        message: userMessage,
        history: messages.slice(-10),
        top_k: topK,
        use_reranker: useReranker
      }
      // Only include patient_id if in patient-specific mode
      if (selectedPatient) {
        body.patient_id = selectedPatient
      }

      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      })

      const data = await res.json()
      
      setMessages(prev => [...prev, { role: 'assistant', content: data.answer }])
      setLastMetrics(data.metrics)
      setLastChunks(data.retrieved_chunks || [])
      setLastToolCalls(data.tool_calls || [])
    } catch (err) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: `Error: ${err.message}. Check if backend is running.` 
      }])
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const toggleChunk = (idx) => {
    setExpandedChunks(prev => ({ ...prev, [idx]: !prev[idx] }))
  }

  const currentPatient = patients.find(p => p.patient_id === selectedPatient)
  const isGlobalMode = selectedPatient === null

  return (
    <div className="app">
      <style>{`
        :root {
          --box-padding: 24px 32px;
          --box-max-width: 900px;
          --name-font: 20px;
          --nav-font: 14px;
          --hint-font: 11px;
          --msg-font: 13px;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
          background: #fff;
          font-family: "Courier New", Courier, monospace;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: flex-start;
          min-height: 100vh;
          padding: 20px;
        }
        
        .app {
          width: 100%;
          max-width: var(--box-max-width);
          display: flex;
          flex-direction: column;
          align-items: center;
        }
        
        .box {
          border: 1px solid #000;
          padding: var(--box-padding);
          width: 100%;
          margin-bottom: 20px;
        }
        
        .header {
          text-align: center;
          margin-bottom: 16px;
          padding-bottom: 12px;
          border-bottom: 1px solid #ccc;
        }
        
        .title {
          font-size: var(--name-font);
          font-weight: bold;
          color: #333;
        }
        
        .subtitle {
          font-size: 12px;
          color: #666;
          margin-top: 4px;
        }
        
        .stats {
          font-size: 11px;
          color: #888;
          margin-top: 8px;
        }
        
        .warning {
          font-size: 10px;
          color: #b45309;
          background: #fef3c7;
          padding: 4px 8px;
          margin-top: 8px;
          display: inline-block;
        }
        
        .nav {
          list-style: none;
          margin-bottom: 16px;
        }
        
        .nav li {
          padding: 6px 10px;
          margin: 2px 0;
          font-size: var(--nav-font);
          display: flex;
          align-items: center;
          justify-content: space-between;
          cursor: pointer;
        }
        
        .nav li:hover { background: #f0f0f0; }
        .nav li.selected { background: #0000EE; color: #fff; }
        .nav li.selected .mode-badge { background: #fff; color: #0000EE; }
        
        .mode-badge {
          font-size: 10px;
          padding: 2px 6px;
          background: #0000EE;
          color: #fff;
        }
        
        .mode-badge.active {
          background: #00aa44;
          color: #fff;
        }
        
        .nav li.selected .mode-badge.active {
          background: #00ff66;
          color: #000;
        }
        
        .patient-selector {
          padding: 12px;
          background: #f8f8f8;
          border: 1px solid #ddd;
          margin-bottom: 16px;
        }
        
        .patient-selector select {
          width: 100%;
          padding: 6px;
          font-family: inherit;
          font-size: 12px;
          border: 1px solid #ccc;
          margin-top: 8px;
        }
        
        .patient-info {
          font-size: 11px;
          color: #666;
          margin-top: 8px;
          padding-top: 8px;
          border-top: 1px solid #ddd;
        }
        
        .settings-row {
          display: flex;
          gap: 16px;
          margin-top: 8px;
          font-size: 11px;
        }
        
        .settings-row label {
          display: flex;
          align-items: center;
          gap: 4px;
        }
        
        .chat-box {
          border: 1px solid #ccc;
          height: 300px;
          overflow-y: auto;
          padding: 12px;
          margin-bottom: 12px;
          background: #fafafa;
        }
        
        .empty-state {
          text-align: center;
          color: #888;
          padding: 40px 20px;
          font-size: 12px;
        }
        
        .message {
          margin-bottom: 12px;
          font-size: var(--msg-font);
        }
        
        .message.user {
          text-align: right;
        }
        
        .message.user .content {
          display: inline-block;
          background: #0000EE;
          color: #fff;
          padding: 8px 12px;
          max-width: 80%;
          text-align: left;
        }
        
        .message.assistant .content {
          background: #f0f0f0;
          padding: 8px 12px;
          border-left: 2px solid #0000EE;
        }
        
        .message.assistant .content p { margin: 0 0 8px 0; }
        .message.assistant .content p:last-child { margin: 0; }
        .message.assistant .content ul, 
        .message.assistant .content ol { margin: 0 0 8px 16px; }
        .message.assistant .content strong { color: #333; }
        
        .loading {
          color: #666;
          font-style: italic;
        }
        
        .input-row {
          display: flex;
          gap: 8px;
        }
        
        .input-row input {
          flex: 1;
          padding: 8px 12px;
          font-family: inherit;
          font-size: 13px;
          border: 1px solid #000;
        }
        
        .input-row input:focus {
          outline: 2px solid #0000EE;
          outline-offset: -2px;
        }
        
        .input-row button {
          padding: 8px 16px;
          font-family: inherit;
          font-size: 13px;
          background: #000;
          color: #fff;
          border: none;
          cursor: pointer;
        }
        
        .input-row button:hover { background: #333; }
        .input-row button:disabled { background: #ccc; cursor: not-allowed; }
        
        .hint {
          font-size: var(--hint-font);
          color: #666;
          margin-top: 16px;
          padding-top: 12px;
          border-top: 1px solid #ccc;
          text-align: center;
        }
        
        .evidence-box {
          margin-top: 16px;
          padding-top: 16px;
          border-top: 1px solid #ccc;
        }
        
        .evidence-title {
          font-size: 12px;
          font-weight: bold;
          margin-bottom: 8px;
        }
        
        .metrics {
          display: flex;
          gap: 16px;
          font-size: 10px;
          color: #666;
          margin-bottom: 12px;
        }
        
        .metrics span {
          background: #f0f0f0;
          padding: 2px 6px;
        }
        
        .tool-calls {
          font-size: 10px;
          margin-bottom: 12px;
        }
        
        .tool-call {
          background: #f5f0ff;
          padding: 4px 8px;
          margin: 4px 0;
          border-left: 2px solid #8b5cf6;
        }
        
        .chunks {
          max-height: 200px;
          overflow-y: auto;
        }
        
        .chunk {
          font-size: 11px;
          padding: 8px;
          margin: 4px 0;
          background: #f8f8f8;
          border: 1px solid #e0e0e0;
          cursor: pointer;
        }
        
        .chunk:hover { background: #f0f0f0; }
        
        .chunk-header {
          display: flex;
          justify-content: space-between;
          margin-bottom: 4px;
        }
        
        .chunk-type {
          color: #0000EE;
          font-weight: bold;
        }
        
        .chunk-score {
          color: #00aa44;
        }
        
        .chunk-patient {
          color: #666;
          font-size: 10px;
        }
        
        .chunk-text {
          color: #333;
          line-height: 1.4;
        }
        
        .chunk-text.collapsed {
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }
      `}</style>

      <div className="box">
        {/* Header */}
        <div className="header">
          <div className="title">~/ehr-rag-agent/</div>
          <div className="subtitle">
            {isGlobalMode ? 'GLOBAL SEARCH MODE' : `Patient: ${selectedPatient}`}
          </div>
          {stats && (
            <div className="stats">
              {stats.patients} patients | {stats.documents} docs | {stats.chunks_indexed} chunks
            </div>
          )}
          <div className="warning">DEMO ONLY - synthetic data - not for clinical use</div>
        </div>

        {/* Navigation */}
        <ul className="nav">
          {menuItems.map((item, i) => (
            <li 
              key={i}
              className={i === menuIdx ? 'selected' : ''}
              onClick={() => {
                setMenuIdx(i)
                item.action()
              }}
            >
              <span>{item.label}</span>
              {item.label === 'global search' && isGlobalMode && (
                <span className="mode-badge active">active</span>
              )}
              {item.label === 'select patient' && !isGlobalMode && (
                <span className="mode-badge active">{selectedPatient}</span>
              )}
              {item.label === 'evidence' && lastChunks.length > 0 && (
                <span className="mode-badge">{lastChunks.length} chunks</span>
              )}
            </li>
          ))}
        </ul>

        {/* Patient Selector (collapsible) */}
        {showSettings && (
          <div className="patient-selector">
            <strong>Select Patient (or leave empty for global):</strong>
            <select
              value={selectedPatient || ''}
              onChange={(e) => {
                const val = e.target.value
                setSelectedPatient(val || null)
                setMessages([])
                setLastChunks([])
                setLastToolCalls([])
              }}
            >
              <option value="">-- Global Search (all patients) --</option>
              {patients.map(p => (
                <option key={p.patient_id} value={p.patient_id}>
                  {p.patient_id} | {p.age}{p.sex} | {p.primary_diagnosis?.substring(0, 40)}
                </option>
              ))}
            </select>
            
            {currentPatient && (
              <div className="patient-info">
                <strong>{currentPatient.patient_id}</strong> - {currentPatient.age}yo {currentPatient.sex}<br/>
                Dx: {currentPatient.primary_diagnosis}<br/>
                {currentPatient.disease_stage && `Stage: ${currentPatient.disease_stage}`}
              </div>
            )}

            <div className="settings-row">
              <label>
                Top-K:
                <input
                  type="range"
                  min="1"
                  max="15"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  style={{ width: 60 }}
                />
                <span>{topK}</span>
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={useReranker}
                  onChange={(e) => setUseReranker(e.target.checked)}
                />
                Reranker
              </label>
            </div>
          </div>
        )}

        {/* Chat Area */}
        <div className="chat-box">
          {messages.length === 0 ? (
            <div className="empty-state">
              {isGlobalMode 
                ? 'Global mode: Ask questions across all 100 patients'
                : `Patient mode: Ask about ${selectedPatient}`
              }
              <br/><br/>
              Examples:<br/>
              {isGlobalMode ? (
                <>
                  "How many patients have diabetes?"<br/>
                  "Find patients with abnormal liver function"<br/>
                  "Which patients are on anticoagulants?"
                </>
              ) : (
                <>
                  "Summarize this patient's condition"<br/>
                  "Check for medication contraindications"<br/>
                  "What imaging findings support the diagnosis?"
                </>
              )}
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx} className={`message ${msg.role}`}>
                <div className="content">
                  {msg.role === 'assistant' ? (
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  ) : (
                    msg.content
                  )}
                </div>
              </div>
            ))
          )}
          {loading && (
            <div className="message assistant">
              <div className="content loading">Searching and analyzing...</div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="input-row">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={isGlobalMode ? "Ask about all patients..." : `Ask about ${selectedPatient}...`}
            disabled={loading}
          />
          <button onClick={sendMessage} disabled={loading || !input.trim()}>
            {loading ? '...' : 'send'}
          </button>
        </div>

        {/* Evidence Panel (collapsible) */}
        {showEvidence && (lastMetrics || lastChunks.length > 0) && (
          <div className="evidence-box">
            <div className="evidence-title">Evidence & Trace</div>
            
            {lastMetrics && (
              <div className="metrics">
                <span>total: {lastMetrics.total_latency_ms}ms</span>
                <span>retrieval: {lastMetrics.retrieval_latency_ms}ms</span>
                <span>llm: {lastMetrics.llm_latency_ms}ms</span>
                <span>tokens: {lastMetrics.tokens_in}->{lastMetrics.tokens_out}</span>
              </div>
            )}

            {lastToolCalls.length > 0 && (
              <div className="tool-calls">
                {lastToolCalls.map((tc, idx) => (
                  <div key={idx} className="tool-call">
                    <strong>{tc.tool_name}</strong> ({tc.latency_ms}ms)<br/>
                    {tc.result_summary}
                  </div>
                ))}
              </div>
            )}

            <div className="chunks">
              {lastChunks.map((chunk, idx) => (
                <div 
                  key={idx} 
                  className="chunk"
                  onClick={() => toggleChunk(idx)}
                >
                  <div className="chunk-header">
                    <span>
                      <span className="chunk-type">{chunk.doc_type}</span>
                      {chunk.patient_id && chunk.patient_id !== selectedPatient && (
                        <span className="chunk-patient"> [{chunk.patient_id}]</span>
                      )}
                    </span>
                    <span className="chunk-score">{(chunk.score * 100).toFixed(1)}%</span>
                  </div>
                  <div className={`chunk-text ${expandedChunks[idx] ? '' : 'collapsed'}`}>
                    {chunk.text}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="hint">
          j/k or arrows to navigate | enter to select | / to focus input | esc to blur
        </div>
      </div>
    </div>
  )
}

export default App
