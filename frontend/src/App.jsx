import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'

// API base URL
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
  const [useReranker, setUseReranker] = useState(true)
  const [lastMetrics, setLastMetrics] = useState(null)
  const [lastChunks, setLastChunks] = useState([])
  const [lastToolCalls, setLastToolCalls] = useState([])
  const [expandedChunks, setExpandedChunks] = useState({})
  const [stats, setStats] = useState(null)
  
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

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (document.activeElement === inputRef.current) return
      if (e.key === '/' || e.key === 'i') {
        e.preventDefault()
        inputRef.current?.focus()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

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
          --name-font: 18px;
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
        
        .stats {
          font-size: 11px;
          color: #888;
          margin-top: 6px;
        }
        
        .warning {
          font-size: 10px;
          color: #b45309;
          background: #fef3c7;
          padding: 4px 8px;
          margin-top: 8px;
          display: inline-block;
        }
        
        .controls {
          margin-bottom: 16px;
          padding-bottom: 16px;
          border-bottom: 1px solid #ccc;
        }
        
        .patient-select-row {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 8px;
        }
        
        .patient-select-row label {
          font-size: 12px;
          color: #666;
        }
        
        .patient-select-row select {
          flex: 1;
          padding: 6px 8px;
          font-family: inherit;
          font-size: 12px;
          border: 1px solid #ccc;
        }
        
        .mode-indicator {
          font-size: 10px;
          padding: 2px 6px;
          background: #e0e0e0;
          color: #666;
        }
        
        .mode-indicator.global {
          background: #dbeafe;
          color: #1d4ed8;
        }
        
        .mode-indicator.patient {
          background: #dcfce7;
          color: #166534;
        }
        
        .patient-info {
          font-size: 11px;
          color: #666;
          padding: 8px;
          background: #f8f8f8;
          border: 1px solid #e0e0e0;
        }
        
        .settings-row {
          display: flex;
          gap: 16px;
          margin-top: 8px;
          font-size: 11px;
          color: #666;
        }
        
        .settings-row label {
          display: flex;
          align-items: center;
          gap: 4px;
        }
        
        .chat-area {
          display: flex;
          gap: 16px;
        }
        
        .chat-main {
          flex: 2;
        }
        
        .chat-box {
          border: 1px solid #ccc;
          height: 280px;
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
        
        .evidence-panel {
          flex: 1;
          min-width: 280px;
        }
        
        .evidence-section {
          margin-bottom: 12px;
        }
        
        .evidence-title {
          font-size: 11px;
          font-weight: bold;
          color: #666;
          margin-bottom: 6px;
          text-transform: uppercase;
        }
        
        .metrics {
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
          font-size: 10px;
        }
        
        .metrics span {
          background: #f0f0f0;
          padding: 2px 6px;
          color: #666;
        }
        
        .tool-calls {
          font-size: 10px;
        }
        
        .tool-call {
          background: #f5f0ff;
          padding: 4px 8px;
          margin: 4px 0;
          border-left: 2px solid #8b5cf6;
        }
        
        .chunks-list {
          max-height: 200px;
          overflow-y: auto;
        }
        
        .chunk {
          font-size: 10px;
          padding: 6px;
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
          font-size: 9px;
        }
        
        .chunk-text {
          color: #333;
          line-height: 1.3;
        }
        
        .chunk-text.collapsed {
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }
        
        .hint {
          font-size: var(--hint-font);
          color: #888;
          margin-top: 16px;
          padding-top: 12px;
          border-top: 1px solid #ccc;
          text-align: center;
        }
      `}</style>

      <div className="box">
        {/* Header */}
        <div className="header">
          <div className="title">~/clinical-rag-agent/</div>
          {stats && (
            <div className="stats">
              {stats.patients} patients | {stats.documents} docs | {stats.chunks_indexed} vectors
            </div>
          )}
          <div className="warning">synthetic data only - not for clinical use</div>
        </div>

        {/* Controls */}
        <div className="controls">
          <div className="patient-select-row">
            <label>Patient:</label>
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
              <option value="">Global (all patients)</option>
              {patients.map(p => (
                <option key={p.patient_id} value={p.patient_id}>
                  {p.patient_id} | {p.age}{p.sex} | {p.primary_diagnosis?.substring(0, 35)}
                </option>
              ))}
            </select>
            <span className={`mode-indicator ${isGlobalMode ? 'global' : 'patient'}`}>
              {isGlobalMode ? 'global' : selectedPatient}
            </span>
          </div>
          
          {currentPatient && (
            <div className="patient-info">
              <strong>{currentPatient.patient_id}</strong> - {currentPatient.age}yo {currentPatient.sex} - {currentPatient.primary_diagnosis}
              {currentPatient.disease_stage && ` (${currentPatient.disease_stage})`}
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

        {/* Chat + Evidence */}
        <div className="chat-area">
          {/* Chat */}
          <div className="chat-main">
            <div className="chat-box">
              {messages.length === 0 ? (
                <div className="empty-state">
                  {isGlobalMode 
                    ? 'Global mode: query across all patients\n\n"How many patients have diabetes?"\n"Find patients on anticoagulants"'
                    : `Patient mode: ${selectedPatient}\n\n"Summarize condition"\n"Check for contraindications"`
                  }
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
                  <div className="content loading">analyzing...</div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            <div className="input-row">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={isGlobalMode ? "query all patients..." : `query ${selectedPatient}...`}
                disabled={loading}
              />
              <button onClick={sendMessage} disabled={loading || !input.trim()}>
                {loading ? '...' : 'send'}
              </button>
            </div>
          </div>

          {/* Evidence Panel - always visible */}
          <div className="evidence-panel">
            {/* Metrics */}
            {lastMetrics && (
              <div className="evidence-section">
                <div className="evidence-title">Metrics</div>
                <div className="metrics">
                  <span>{lastMetrics.total_latency_ms}ms total</span>
                  <span>{lastMetrics.retrieval_latency_ms}ms retrieval</span>
                  <span>{lastMetrics.llm_latency_ms}ms llm</span>
                  <span>{lastMetrics.tokens_in}+{lastMetrics.tokens_out} tokens</span>
                </div>
              </div>
            )}

            {/* Tool Calls */}
            {lastToolCalls.length > 0 && (
              <div className="evidence-section">
                <div className="evidence-title">Tools ({lastToolCalls.length})</div>
                <div className="tool-calls">
                  {lastToolCalls.map((tc, idx) => (
                    <div key={idx} className="tool-call">
                      <strong>{tc.tool_name}</strong> ({tc.latency_ms}ms)<br/>
                      {tc.result_summary}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Chunks */}
            <div className="evidence-section">
              <div className="evidence-title">Retrieved ({lastChunks.length})</div>
              <div className="chunks-list">
                {lastChunks.length === 0 ? (
                  <div style={{ fontSize: 10, color: '#888' }}>no chunks yet</div>
                ) : (
                  lastChunks.map((chunk, idx) => (
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
                        <span className="chunk-score">{(chunk.score * 100).toFixed(0)}%</span>
                      </div>
                      <div className={`chunk-text ${expandedChunks[idx] ? '' : 'collapsed'}`}>
                        {chunk.text}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="hint">/ to focus input</div>
      </div>
    </div>
  )
}

export default App
