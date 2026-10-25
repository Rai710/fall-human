export type AlertRow = {
  alert_id?: string
  track_id?: string
  frame?: string | number
  timestamp_ms?: string | number
  label?: string
  confidence?: string | number
  is_fall?: string | number
  fall_streak?: string | number
  segment_id?: string | number
  fall_score?: string | number
}

export type LiveStats = {
  frame?: number
  timestamp_ms?: number
  total_frames?: number
  alerts_count?: number
  inference_fps?: number
  analyses?: number
  device?: string
  label?: string
  confidence?: number
  fall_streak?: number
  is_fall?: boolean
  infer_busy?: boolean
  last_result_age_ms?: number | null
  tracked_people?: number
  detections?: number
  pending_people?: number
  detector_ms?: number | null
  tracking_stale?: boolean
  active_alarms?: number
}

export type DetectorInfo = {
  source: 'pretrained' | 'local' | 'huggingface'
  label: string
  ready: boolean
  classes: number[] | null
  min_confidence: number
}

export type Summary = {
  status?: string
  source?: string
  model?: string
  device?: string
  frames_processed?: number
  analyses?: number
  alerts_count?: number
  warning?: string | null
  [key: string]: unknown
}

export type Snapshot = {
  id: string
  state: string
  error: string | null
  message?: string
  source: string
  name?: string
  output: string | null
  stats: LiveStats
  alerts: AlertRow[]
  summary: Summary | null
  started_at: number
  finished_at: number | null
}

export type SessionListItem = {
  id: string
  modified_at: number
  source?: string | null
  model?: string | null
  device?: string | null
  status?: string | null
  frames_processed: number
  alerts_count: number
  acknowledged: number
  warning?: string | null
  has_video: boolean
}

export type SessionDetail = {
  id: string
  summary: Summary
  alerts: AlertRow[]
  predictions: AlertRow[]
  review: { acked: string[]; labels: Record<string, string> }
  files: string[]
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    let message = `Permintaan gagal (HTTP ${response.status}).`
    try {
      const data = (await response.json()) as { detail?: unknown }
      if (typeof data.detail === 'string') message = data.detail
    } catch {
      // keep default message
    }
    throw new Error(message)
  }
  return (await response.json()) as T
}

export function startSession(form: FormData): Promise<Snapshot> {
  return request<Snapshot>('/api/sessions/start', { method: 'POST', body: form })
}

export function stopSession(id: string): Promise<Snapshot> {
  return request<Snapshot>(`/api/sessions/${id}/stop`, { method: 'POST' })
}

export function getSession(id: string): Promise<SessionDetail> {
  return request<SessionDetail>(`/api/sessions/${id}`)
}

export function getConfig(): Promise<{ detector: DetectorInfo }> {
  return request<{ detector: DetectorInfo }>('/api/config')
}

export function listSessions(): Promise<SessionListItem[]> {
  return request<SessionListItem[]>('/api/sessions')
}

export function saveReview(
  id: string,
  acked: string[],
  labels?: Record<string, string>,
): Promise<{ acked: string[]; labels: Record<string, string> }> {
  return request(`/api/sessions/${id}/review`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(labels ? { acked, labels } : { acked }),
  })
}

export function streamUrl(id: string): string {
  return `/api/sessions/${id}/stream`
}

export function eventsUrl(id: string): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}/api/sessions/${id}/events`
}

export function fileUrl(id: string, filename: string): string {
  return `/api/sessions/${id}/files/${filename}`
}

export function downloadUrl(id: string): string {
  return `/api/sessions/${id}/download`
}
