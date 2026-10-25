import { useEffect, useState } from 'react'
import {
  downloadUrl,
  fileUrl,
  getSession,
  listSessions,
  saveReview,
  type AlertRow,
  type SessionDetail,
  type SessionListItem,
} from './api'

type Props = {
  focusId: string | null
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

function formatDate(seconds: number): string {
  return new Date(seconds * 1000).toLocaleString('id-ID')
}

function alertKey(alert: AlertRow, index: number): string {
  return String(alert.alert_id ?? alert.track_id ?? index)
}

function sourceLabel(source?: string | null): string {
  if (!source) return '—'
  if (/^\d+$/.test(source)) return `Webcam ${source}`
  return source.split(/[\\/]/).pop() || source
}

export default function History({ focusId }: Props) {
  const [sessions, setSessions] = useState<SessionListItem[]>([])
  const [selected, setSelected] = useState<string | null>(focusId)
  const [detail, setDetail] = useState<SessionDetail | null>(null)
  const [acked, setAcked] = useState<Set<string>>(new Set())
  const [labels, setLabels] = useState<Record<string, string>>({})
  const [loading, setLoading] = useState(Boolean(focusId))
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState('')

  async function refresh() {
    try {
      setSessions(await listSessions())
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : 'Gagal memuat riwayat.')
    }
  }

  useEffect(() => {
    let cancelled = false
    listSessions()
      .then((data) => {
        if (!cancelled) setSessions(data)
      })
      .catch((exc: unknown) => {
        if (!cancelled) setError(exc instanceof Error ? exc.message : 'Gagal memuat riwayat.')
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!selected) return
    let cancelled = false
    getSession(selected)
      .then((data) => {
        if (cancelled) return
        setDetail(data)
        setAcked(new Set(data.review.acked))
        setLabels(data.review.labels)
        setSaved(false)
        setError('')
      })
      .catch((exc: unknown) => {
        if (cancelled) return
        setError(exc instanceof Error ? exc.message : 'Gagal memuat detail sesi.')
        setDetail(null)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [selected])

  function selectSession(id: string) {
    if (id === selected) return
    setSelected(id)
    setDetail(null)
    setLoading(true)
    setError('')
  }

  function toggleAck(key: string) {
    setAcked((previous) => {
      const next = new Set(previous)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
    setSaved(false)
  }

  function updateLabel(key: string, value: string) {
    setLabels((previous) => ({ ...previous, [key]: value }))
    setSaved(false)
  }

  async function handleSave() {
    if (!detail) return
    setSaving(true)
    setError('')
    try {
      const review = await saveReview(detail.id, Array.from(acked), labels)
      setDetail({ ...detail, review })
      setSaved(true)
      await refresh()
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : 'Gagal menyimpan tinjauan.')
    } finally {
      setSaving(false)
    }
  }

  const predictions = detail ? detail.predictions.slice(-200).reverse() : []

  return (
    <div className="history-layout">
      <aside className="panel session-list">
        <div className="panel-head compact">
          <h2>Riwayat sesi</h2>
          <button type="button" className="ghost small" onClick={() => void refresh()}>
            Muat ulang
          </button>
        </div>
        {error && <div className="error">{error}</div>}
        {sessions.length === 0 ? (
          <p className="muted empty-text">Belum ada sesi tersimpan.</p>
        ) : (
          <ul className="session-items">
            {sessions.map((item) => (
              <li key={item.id}>
                <button
                  type="button"
                  className={selected === item.id ? 'session-item active' : 'session-item'}
                  onClick={() => selectSession(item.id)}
                >
                  <span className="session-title">{sourceLabel(item.source)}</span>
                  <span className="muted">{formatDate(item.modified_at)}</span>
                  <span className="session-tags">
                    <span className="tag">{item.alerts_count} alert</span>
                    <span className="tag">{item.acknowledged} ditandai</span>
                    {item.status && <span className="tag">{item.status}</span>}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </aside>

      <section className="panel session-detail">
        {loading && <p className="muted">Memuat detail...</p>}
        {!loading && !detail && <p className="muted empty-text">Pilih sesi untuk melihat detail.</p>}
        {detail && (
          <>
            <div className="panel-head">
              <div>
                <h2>{detail.id}</h2>
                <p className="muted">{sourceLabel(detail.summary.source)}</p>
              </div>
              <div className="actions inline">
                <a className="button secondary" href={downloadUrl(detail.id)}>
                  Unduh ZIP
                </a>
                <button
                  type="button"
                  className="primary"
                  onClick={handleSave}
                  disabled={saving}
                >
                  {saving ? 'Menyimpan...' : saved ? 'Tersimpan' : 'Simpan tinjauan'}
                </button>
              </div>
            </div>

            {detail.summary.warning && <div className="warning">{String(detail.summary.warning)}</div>}

            <div className="stat-grid">
              <div className="stat">
                <span>Status</span>
                <strong>{detail.summary.status ?? '—'}</strong>
              </div>
              <div className="stat">
                <span>Frame diproses</span>
                <strong>{detail.summary.frames_processed ?? 0}</strong>
              </div>
              <div className="stat">
                <span>Total alert</span>
                <strong>{detail.alerts.length}</strong>
              </div>
              <div className="stat">
                <span>Device</span>
                <strong>{detail.summary.device ?? '—'}</strong>
                <small>{detail.summary.model ?? ''}</small>
              </div>
            </div>

            {detail.files.includes('annotated.mp4') ? (
              <video className="history-video" src={fileUrl(detail.id, 'annotated.mp4')} controls />
            ) : (
              <p className="muted">Video anotasi tidak tersedia untuk sesi webcam.</p>
            )}

            <h3>Alert</h3>
            {detail.alerts.length === 0 ? (
              <p className="muted">Tidak ada alert pada sesi ini.</p>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Waktu</th>
                      <th>Orang</th>
                      <th>Label</th>
                      <th>Keyakinan</th>
                      <th>Ditandai</th>
                      <th>Klasifikasi</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detail.alerts.map((alert, index) => {
                      const key = alertKey(alert, index)
                      return (
                        <tr key={`${key}-${index}`}>
                          <td className="mono">{formatClock(alert.timestamp_ms)}</td>
                          <td className="mono">
                            {alert.track_id ? `${alert.track_id}.${alert.segment_id ?? 0}` : '—'}
                          </td>
                          <td>{alert.label ?? 'fall'}</td>
                          <td className="mono">{formatConfidence(alert.confidence)}</td>
                          <td>
                            <input
                              type="checkbox"
                              checked={acked.has(key)}
                              onChange={() => toggleAck(key)}
                            />
                          </td>
                          <td>
                            <select
                              value={labels[key] ?? ''}
                              onChange={(event) => updateLabel(key, event.target.value)}
                            >
                              <option value="">Belum dinilai</option>
                              <option value="true_fall">Jatuh asli</option>
                              <option value="false_alarm">Alarm palsu</option>
                            </select>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}

            <h3>Prediksi terakhir</h3>
            {predictions.length === 0 ? (
              <p className="muted">Tidak ada data prediksi.</p>
            ) : (
              <div className="table-wrap short">
                <table>
                  <thead>
                    <tr>
                      <th>Waktu</th>
                      <th>Orang</th>
                      <th>Label</th>
                      <th>Keyakinan</th>
                      <th>Streak</th>
                    </tr>
                  </thead>
                  <tbody>
                    {predictions.map((row, index) => (
                      <tr key={index}>
                        <td className="mono">{formatClock(row.timestamp_ms)}</td>
                        <td className="mono">
                          {row.track_id ? `${row.track_id}.${row.segment_id ?? 0}` : '—'}
                        </td>
                        <td>{row.label ?? '—'}</td>
                        <td className="mono">{formatConfidence(row.confidence)}</td>
                        <td className="mono">{row.fall_streak ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </section>
    </div>
  )
}
