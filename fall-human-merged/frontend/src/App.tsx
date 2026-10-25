import { useState } from 'react'
import History from './History'
import Live from './Live'

type Tab = 'live' | 'history'

function App() {
  const [tab, setTab] = useState<Tab>('live')
  const [focusId, setFocusId] = useState<string | null>(null)

  function openHistory(id: string) {
    setFocusId(id)
    setTab('history')
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">FD</span>
          <div>
            <strong>Fall Detection</strong>
            <span>Dashboard monitoring lokal</span>
          </div>
        </div>
        <nav className="tabs">
          <button
            type="button"
            className={tab === 'live' ? 'tab active' : 'tab'}
            onClick={() => {
              setFocusId(null)
              setTab('live')
            }}
          >
            Live
          </button>
          <button
            type="button"
            className={tab === 'history' ? 'tab active' : 'tab'}
            onClick={() => {
              setFocusId(null)
              setTab('history')
            }}
          >
            Riwayat
          </button>
        </nav>
      </header>

      <main>
        {tab === 'live' ? (
          <Live onOpenHistory={openHistory} />
        ) : (
          <History focusId={focusId} />
        )}
      </main>
    </div>
  )
}

export default App
