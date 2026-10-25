import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import {
  eventsUrl,
  fileUrl,
  getConfig,
  saveReview,
  startSession,
  stopSession,
  streamUrl,
  type AlertRow,
  type ActionModelInfo,
  type DetectorInfo,
  type LiveStats,
  type Snapshot,
} from './api'

type Props = {
  onOpenHistory: (id: string) => void
}

type Mode = 'camera' | 'file'

function isActive(state?: string): boolean {
  return state === 'starting' || state === 'running' || state === 'stopping'
}

function statusText(state?: string): string {
  switch (state) {
    case 'starting':
      return 'Memuat model'
    case 'running':
      return 'Berjalan'
    case 'stopping':
      return 'Menghentikan'
    case 'completed':
      return 'Selesai'
    case 'stopped':
      return 'Dihentikan'
    case 'failed':
      return 'Gagal'
    default:
      return 'Siap'
  }
}

function sourceLabel(source?: string): string {
  if (!source) return '—'
  if (/^\d+$/.test(source)) return `Webcam ${source}`
  return source.split(/[\\/]/).pop() || source
}

function formatClock(value?: string | number | null): string {
  const ms = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(ms)) return '—'
  const total = Math.floor(ms / 1000)
  const minutes = Math.floor(total / 60)
  const seconds = total % 60
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
}

function formatConfidence(value?: string | number | null): string {
  const number = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(number)) return '—'
  return `${(number * 100).toFixed(1)}%`
}

const SOURCE_NAME: Record<DetectorInfo['source'], string> = {
  pretrained: 'pretrained',
  local: 'bobot lokal',
  huggingface: 'Hugging Face',
}

function alertKey(alert: AlertRow, index: number): string {
  return String(alert.alert_id ?? alert.track_id ?? index)
}

function playAlarm(): void {
  try {
    const context = new AudioContext()
    const gain = context.createGain()
    gain.gain.setValueAtTime(0.12, context.currentTime)
    gain.gain.exponentialRampToValueAtTime(0.001, context.currentTime + 0.7)
    const oscillator = context.createOscillator()
    oscillator.type = 'sawtooth'
    oscillator.frequency.setValueAtTime(880, context.currentTime)
    oscillator.frequency.setValueAtTime(660, context.currentTime + 0.22)
    oscillator.connect(gain)
    gain.connect(context.destination)
    oscillator.start()
    oscillator.stop(context.currentTime + 0.7)
    oscillator.onended = () => {
      void context.close()
    }
  } catch {
    // audio is optional
  }
}

export default function Live({ onOpenHistory }: Props) {
  const [mode, setMode] = useState<Mode>('file')
  const [cameraIndex, setCameraIndex] = useState('0')
  const [file, setFile] = useState<File | null>(null)
  const [saveVideo, setSaveVideo] = useState(true)
  const [realtime, setRealtime] = useState(false)
  const [analysisScope, setAnalysisScope] = useState<"scene" | "person">("scene")
  const [actionModel, setActionModel] = useState<ActionModelInfo | null>(null)
  const [detector, setDetector] = useState<DetectorInfo | null>(null)
  const [session, setSession] = useState<Snapshot | null>(null)
  const [stats, setStats] = useState<LiveStats>({})
  const [alerts, setAlerts] = useState<AlertRow[]>([])
  const [acked, setAcked] = useState<Set<string>>(new Set())
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [soundOn, setSoundOn] = useState(true)
  const lastAlertCount = useRef(0)
  const sessionId = session?.id
  const sessionState = session?.state
  const active = isActive(sessionState)

  useEffect(() => {
    if (!sessionId || !isActive(sessionState)) return
    const socket = new WebSocket(eventsUrl(sessionId))
    socket.onmessage = (event) => {
      const snapshot = JSON.parse(event.data as string) as Snapshot
      setSession(snapshot)
      setStats(snapshot.stats ?? {})
      setAlerts(snapshot.alerts ?? [])
      if (snapshot.error) setError(snapshot.error)
    }
    socket.onerror = () => {
      setError('Koneksi live terputus. Periksa apakah backend masih berjalan.')
    }
    return () => {
      socket.close()
    }
  }, [sessionId, sessionState])

  useEffect(() => {
    getConfig()
      .then((config) => { setDetector(config.detector); setActionModel(config.action_model) })
      .catch(() => setDetector(null))
  }, [])

  useEffect(() => {
    const count = stats.alerts_count ?? 0
    if (count > lastAlertCount.current && soundOn) playAlarm()
    lastAlertCount.current = count
  }, [stats.alerts_count, soundOn])

  async function handleStart(event: FormEvent) {
    event.preventDefault()
    setError('')
    setStats({})
    setAlerts([])
    setAcked(new Set())
    setSession(null)
    lastAlertCount.current = 0

    if (mode === 'file' && !file) {
      setError('Pilih file video terlebih dahulu.')
      return
    }
    if (mode === 'camera' && cameraIndex.trim() === '') {
      setError('Isi nomor webcam.')
      return
    }

    const form = new FormData()
    form.append("analysis_scope", analysisScope)
    if (mode === 'file' && file) {
      form.append('file', file)
      form.append('save_video', String(saveVideo))
      form.append('realtime', String(realtime))
    } else {
      form.append('camera_index', cameraIndex.trim())
      form.append('save_video', 'false')
    }

    setBusy(true)
    try {
      const snapshot = await startSession(form)
      setSession(snapshot)
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : 'Gagal memulai sesi.')
    } finally {
      setBusy(false)
    }
  }

  async function handleStop() {
    if (!session) return
    setBusy(true)
    setError('')
    try {
      const snapshot = await stopSession(session.id)
      setSession(snapshot)
      setStats(snapshot.stats ?? {})
      setAlerts(snapshot.alerts ?? [])
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : 'Gagal menghentikan sesi.')
    } finally {
      setBusy(false)
    }
  }

  function newSession() {
    setSession(null)
    setStats({})
    setAlerts([])
    setAcked(new Set())
    setError('')
    lastAlertCount.current = 0
  }

  async function persistAck(next: Set<string>) {
    const folder = session?.output
    if (!folder) return
    try {
      await saveReview(folder, Array.from(next))
    } catch {
      // acknowledgement will still be saved from the history page
    }
  }

  function toggleAck(key: string) {
    setAcked((previous) => {
      const next = new Set(previous)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      void persistAck(next)
      return next
    })
  }

  const alertCount = stats.alerts_count ?? alerts.length
  const isCamera = Boolean(session?.summary?.source?.startsWith('webcam:')) || /^\d+$/.test(session?.source ?? '')
  const fallNow = active && Boolean(stats.is_fall)

  return (
    <div className="live-layout">
      <section className="panel">
        <div className="panel-head">
          <div>
            <h1>Live Monitoring</h1>
            <p className="muted">
              Analisis dugaan jatuh dengan VideoMAE. Pilih seluruh kamera atau per orang memakai YOLO + ByteTrack.
            </p>
            {detector && (
              <p className="muted">
                Detektor: <span className="mono">{detector.label}</span> ({SOURCE_NAME[detector.source]})
                {detector.source !== 'huggingface' && !detector.ready
                  ? ' — akan disiapkan saat sesi dimulai'
                  : ''}
              </p>
            )}
          </div>
          <span className={`badge ${session?.state ?? 'idle'}`}>{statusText(session?.state)}</span>
        </div>

        {error && <div className="error">{error}</div>}

        {!session ? (
          <form className="start-form" onSubmit={handleStart}>
            <div className="segmented">
              <button
                type="button"
                className={mode === 'camera' ? 'active' : ''}
                onClick={() => setMode('camera')}
              >
                Webcam
              </button>
              <button
                type="button"
                className={mode === 'file' ? 'active' : ''}
                onClick={() => setMode('file')}
              >
                Upload video
              </button>
            </div>

            {mode === 'camera' ? (
              <label className="field">
                Nomor webcam
                <input
                  type="number"
                  min="0"
                  value={cameraIndex}
                  onChange={(event) => setCameraIndex(event.target.value)}
                />
                <small>Webcam dibuka oleh backend di mesin server.</small>
              </label>
            ) : (
              <>
                <label className="field">
                  File video
                  <input
                    type="file"
                    accept="video/*"
                    onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                  />
                </label>
                <label className="check">
                  <input
                    type="checkbox"
                    checked={saveVideo}
                    onChange={(event) => setSaveVideo(event.target.checked)}
                  />
                  Simpan video anotasi
                </label>
                <label className="check">
                  <input
                    type="checkbox"
                    checked={realtime}
                    onChange={(event) => setRealtime(event.target.checked)}
                  />
                  Ikuti waktu video (simulasi live)
                </label>
              </>
            )}

            <label className="field">
              Area analisis
              <select value={analysisScope} onChange={(event) => setAnalysisScope(event.target.value as "scene" | "person")}>
                <option value="scene">Seluruh kamera — sesuai training model</option>
                <option value="person">Per orang — YOLO + ByteTrack (eksperimen)</option>
              </select>
              <small>{analysisScope === "person"
                ? "Crop per orang mengubah input training; akurasinya perlu dievaluasi tersendiri."
                : "Keputusan berlaku untuk adegan. Mode ini tidak menentukan orang mana yang jatuh."}</small>
            </label>
            <p className="muted">
              Model: {actionModel?.label ?? "Le2i + CAUCAFall (2 kelas)"}
              {actionModel?.threshold != null ? ` · ambang ${(actionModel.threshold * 100).toFixed(0)}%` : ""}
              {actionModel && !actionModel.ready ? " · paket model belum lengkap" : ""}
            </p>
            <button type="submit" className="primary" disabled={busy}>
              {busy ? 'Memulai...' : 'Mulai monitoring'}
            </button>
          </form>
        ) : (
          <>
            <div className="live-meta">
              <span className="mono">{sourceLabel(session.source)}</span>
              <span className="mono">{session.output ?? 'menyiapkan folder...'}</span>
              <label className="check inline">
                <input
                  type="checkbox"
                  checked={soundOn}
                  onChange={(event) => setSoundOn(event.target.checked)}
                />
                Suara alarm
              </label>
            </div>

            <p className="muted">
              {session.message
                ? session.message
                : `${stats.device === 'cpu' ? 'Inferensi CPU. ' : ''}${
                    stats.tracking_stale ? 'Tracking terlambat. ' : ''
                  }${
                    stats.infer_busy
                      ? 'Model sedang menganalisis...'
                      : stats.last_result_age_ms != null
                        ? `Hasil terakhir ${(stats.last_result_age_ms / 1000).toFixed(1)} detik lalu.`
                        : 'Menunggu klip pertama...'
                  }`}
            </p>

            <div className="video-stage">
              {active ? (
                <img className="live-video" src={streamUrl(session.id)} alt="Live stream" />
              ) : !isCamera && session.output ? (
                <video
                  className="live-video"
                  src={fileUrl(session.output, 'annotated.mp4')}
                  controls
                />
              ) : (
                <div className="video-empty">
                  <strong>Sesi selesai</strong>
                  <span>Hasil tersimpan di folder {session.output ?? 'outputs'}.</span>
                </div>
              )}
              {fallNow && <div className="alarm-banner">FALL TERDETEKSI</div>}
            </div>

            <div className="stat-grid">
              <div className="stat">
                <span>Frame</span>
                <strong>{stats.frame ?? 0}</strong>
              </div>
              <div className="stat">
                <span>Pelacakan orang</span>
                <strong>{session.analysis_scope === 'scene' ? 'Tidak aktif' : stats.tracked_people ?? 0}</strong>
                <small>
                  {session.analysis_scope === 'scene' ? 'Mode seluruh kamera' :
                    stats.detector_ms != null ? `deteksi ${stats.detector_ms.toFixed(0)} ms` : 'menunggu deteksi'}
                </small>
              </div>
              <div className="stat">
                <span>Analisis</span>
                <strong>{stats.analyses ?? 0}</strong>
              </div>
              <div className="stat">
                <span>Alert</span>
                <strong>{alertCount}</strong>
              </div>
              <div className="stat">
                <span>Frame FPS</span>
                <strong>{stats.inference_fps ? stats.inference_fps.toFixed(1) : '0.0'}</strong>
              </div>
              <div className="stat">
                <span>Prediksi</span>
                <strong>{stats.label || '—'}</strong>
                <small>{stats.label ? formatConfidence(stats.confidence) : 'menunggu analisis'}</small>
              </div>
            </div>

            <div className="actions">
              {active ? (
                <button type="button" className="danger" onClick={handleStop} disabled={busy}>
                  {busy ? 'Menghentikan...' : 'Hentikan monitoring'}
                </button>
              ) : (
                <>
                  <button type="button" className="primary" onClick={newSession}>
                    Sesi baru
                  </button>
                  {session.output && (
                    <button
                      type="button"
                      className="secondary"
                      onClick={() => onOpenHistory(session.output as string)}
                    >
                      Buka riwayat
                    </button>
                  )}
                </>
              )}
            </div>
          </>
        )}
      </section>

      <aside className="panel alerts-panel">
        <div className="panel-head compact">
          <h2>Log alert</h2>
          <span className="muted">{alerts.length}</span>
        </div>
        {alerts.length === 0 ? (
          <p className="muted empty-text">Belum ada alert pada sesi ini.</p>
        ) : (
          <ul className="alert-list">
            {alerts
              .slice()
              .reverse()
              .map((alert, index) => {
                const key = alertKey(alert, index)
                const isAcked = acked.has(key)
                return (
                  <li key={`${key}-${index}`} className={isAcked ? 'alert-item acked' : 'alert-item'}>
                    <div className="alert-main">
                      <span className="dot" />
                      <strong>{alert.label || 'Fall'}</strong>
                      {alert.track_id && (
                        <span className="muted">
                          {alert.evidence_scope === 'last_known_region' ? 'Area terakhir ID' : 'Orang ID'} {alert.track_id}.{alert.segment_id}
                        </span>
                      )}
                      <span className="muted">{formatClock(alert.timestamp_ms)}</span>
                    </div>
                    <div className="alert-meta">
                      <span>{formatConfidence(alert.confidence)}</span>
                      <button
                        type="button"
                        className="ghost small"
                        onClick={() => toggleAck(key)}
                      >
                        {isAcked ? 'Batal' : 'Tandai'}
                      </button>
                    </div>
                  </li>
                )
              })}
          </ul>
        )}
      </aside>
    </div>
  )
}
