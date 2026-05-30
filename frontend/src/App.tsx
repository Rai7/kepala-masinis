import { useMemo, useRef, useState } from 'react'
import './App.css'

type Role = 'public' | 'internal'

type ChatMessage = {
  id: string
  from: 'user' | 'bot'
  text: string
}

type ChatResponse = {
  intent: 'station_query' | 'train_query' | 'city_to_city_query' | 'unknown'
  reply: string
  clarification?: { question: string; options?: string[] } | null
}

function App() {
  const apiBaseUrl = useMemo(() => {
    const v = (import.meta as any).env?.VITE_API_BASE_URL as string | undefined
    return (v && v.trim()) || 'http://localhost:8000'
  }, [])

  const apiWarning = useMemo(() => {
    const host = window.location.hostname
    const isLocalHost = host === 'localhost' || host === '127.0.0.1'
    const usesLocalApi =
      apiBaseUrl.includes('localhost') || apiBaseUrl.includes('127.0.0.1')
    if (!isLocalHost && usesLocalApi) {
      return `UI ini dibuka dari ${host}, tapi API masih mengarah ke ${apiBaseUrl}. Set VITE_API_BASE_URL ke alamat backend yang bisa diakses dari perangkat ini (mis. http://${host}:8000 atau IP LAN mesin backend).`
    }
    return null
  }, [apiBaseUrl])

  const [role, setRole] = useState<Role>('public')
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'm0',
      from: 'bot',
      text: 'Tanya jadwal GAPEKA 2025. Contoh: “kereta apa saja yang berhenti di GMR” atau “cari kereta argo”.\n\nBersumber dari GAPEKA 2025',
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const listRef = useRef<HTMLDivElement | null>(null)

  const scrollToBottom = () => {
    const el = listRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  }

  const sendMessage = async () => {
    const text = input.trim()
    if (!text || loading) return

    setError(null)
    setInput('')
    const userMsg: ChatMessage = {
      id: `u-${Date.now()}`,
      from: 'user',
      text,
    }
    setMessages((prev) => [...prev, userMsg])
    setLoading(true)

    try {
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, '')}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, role }),
      })

      if (!res.ok) {
        const detail = await res.text()
        throw new Error(detail || `HTTP ${res.status}`)
      }

      const data = (await res.json()) as ChatResponse
      const botMsg: ChatMessage = {
        id: `b-${Date.now()}`,
        from: 'bot',
        text: data.reply,
      }
      setMessages((prev) => [...prev, botMsg])
      queueMicrotask(scrollToBottom)
    } catch (e: any) {
      setError(
        e?.message ||
          `Failed to fetch. Pastikan backend aktif dan VITE_API_BASE_URL benar (saat ini: ${apiBaseUrl}).`
      )
    } finally {
      setLoading(false)
      queueMicrotask(scrollToBottom)
    }
  }

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    void sendMessage()
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <img 
            src="https://coresg-normal.trae.ai/api/ide/v1/text_to_image?prompt=a%20flat%20vector%20icon%20of%20a%20modern%20high-speed%20train%20in%20blue%20inside%20a%20circle%20background%2C%20minimalist&image_size=square" 
            alt="Kepala Stasiun Logo" 
            className="brandLogo" 
          />
          <div className="brandTitle">Kepala Stasiun</div>
        </div>
        <div className="controls">
          <label className="field">
            <span>Role</span>
            <select
              value={role}
              onChange={(e) => setRole(e.target.value as Role)}
              aria-label="Role"
            >
              <option value="public">public</option>
              <option value="internal">internal</option>
            </select>
          </label>
          <div className="apiHint" title="Backend API base URL">
            {apiBaseUrl}
          </div>
        </div>
      </header>

      <main className="chat">
        <div className="chatList" ref={listRef} role="log" aria-live="polite">
          {messages.map((m) => (
            <div
              key={m.id}
              className={m.from === 'user' ? 'bubble bubbleUser' : 'bubble bubbleBot'}
            >
              <pre className="bubbleText">{m.text}</pre>
            </div>
          ))}
          {loading ? (
            <div className="bubble bubbleBot">
              <div className="typing">Memproses…</div>
            </div>
          ) : null}
        </div>

        <form className="composer" onSubmit={onSubmit}>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Tulis pertanyaan…"
            className="composerInput"
            disabled={loading}
          />
          <button className="composerButton" type="submit" disabled={loading || !input.trim()}>
            Kirim
          </button>
        </form>

        {apiWarning ? <div className="warning">{apiWarning}</div> : null}
        {error ? <div className="error">{error}</div> : null}
      </main>
    </div>
  )
}

export default App
