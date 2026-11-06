import { useCallback, useEffect, useMemo, useState } from 'react'
import './App.css'

const EMPTY_READING = {
  product_type: 'M',
  air_temperature_k: '300.0',
  process_temperature_k: '310.0',
  rotational_speed_rpm: '1500',
  torque_nm: '40.0',
  tool_wear_min: '10',
}

const NAV_ITEMS = [
  { id: 'overview', label: 'Overview', icon: '◫' },
  { id: 'sensors', label: 'Sensor readings', icon: '⌁' },
  { id: 'analysis', label: 'Model analysis', icon: '◎' },
  { id: 'investigation', label: 'Investigation', icon: '⌕' },
  { id: 'maintenance', label: 'Maintenance review', icon: '✓' },
]
const TODAY_LABEL = new Intl.DateTimeFormat(undefined, { dateStyle: 'full' }).format(new Date())
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.replace(/\/+$/, '')

class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

async function apiRequest(path, options = {}) {
  const apiUrl = API_BASE_URL
    ? `${API_BASE_URL}${path}`
    : import.meta.env.DEV
      ? `/api${path}`
      : null
  if (!apiUrl) {
    throw new ApiError('API URL is not configured. Set VITE_API_BASE_URL in the deployment settings.', 0)
  }

  let response
  try {
    response = await fetch(apiUrl, {
      ...options,
      headers: {
        ...(options.body ? { 'Content-Type': 'application/json' } : {}),
        ...options.headers,
      },
    })
  } catch {
    const message = import.meta.env.DEV
      ? 'Cannot reach the API. Make sure FastAPI is running on port 8001.'
      : 'Cannot reach the API. Check the configured API URL and backend status.'
    throw new ApiError(message, 0)
  }

  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = Array.isArray(body.detail)
      ? body.detail.map((item) => item.msg).join(', ')
      : body.detail
    throw new ApiError(detail || `Request failed (${response.status}).`, response.status)
  }
  return body
}

function formatDate(value) {
  if (!value) return '—'
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value))
}

function formatNumber(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  return Number(value).toFixed(digits)
}

function IconMark({ children, className = '' }) {
  return <span aria-hidden="true" className={`icon-mark ${className}`}>{children}</span>
}

function EmptyState({ title, children }) {
  return (
    <div className="empty-state">
      <IconMark className="empty-icon">⌁</IconMark>
      <strong>{title}</strong>
      <p>{children}</p>
    </div>
  )
}

function TrendChart({ readings }) {
  const ordered = [...readings].reverse()
  if (ordered.length < 2) {
    return (
      <EmptyState title="Not enough readings yet">
        Save or analyze at least two readings to see a temperature trend.
      </EmptyState>
    )
  }

  const allValues = ordered.flatMap((item) => [
    item.air_temperature_k,
    item.process_temperature_k,
  ])
  const minimum = Math.min(...allValues) - 1
  const maximum = Math.max(...allValues) + 1
  const pointFor = (value, index) => {
    const x = 26 + (index * 548) / Math.max(ordered.length - 1, 1)
    const y = 142 - ((value - minimum) / Math.max(maximum - minimum, 1)) * 112
    return `${x},${y}`
  }

  return (
    <div className="trend-chart">
      <div className="chart-legend">
        <span><i className="legend-dot air" /> Air temperature</span>
        <span><i className="legend-dot process" /> Process temperature</span>
      </div>
      <svg viewBox="0 0 600 170" role="img" aria-label="Recent air and process temperature trend">
        {[30, 66, 102, 138].map((y) => (
          <line key={y} x1="20" x2="580" y1={y} y2={y} className="chart-grid" />
        ))}
        <polyline
          points={ordered.map((item, index) => pointFor(item.air_temperature_k, index)).join(' ')}
          className="chart-line air-line"
        />
        <polyline
          points={ordered.map((item, index) => pointFor(item.process_temperature_k, index)).join(' ')}
          className="chart-line process-line"
        />
      </svg>
      <div className="chart-caption">
        <span>Oldest</span>
        <span>Most recent</span>
      </div>
    </div>
  )
}

function App() {
  const [activePage, setActivePage] = useState('overview')
  const [machineId, setMachineId] = useState('demo-machine-001')
  const [reading, setReading] = useState(EMPTY_READING)
  const [machines, setMachines] = useState([])
  const [history, setHistory] = useState([])
  const [analysis, setAnalysis] = useState(null)
  const [maintenance, setMaintenance] = useState(null)
  const [apiStatus, setApiStatus] = useState('checking')
  const [loading, setLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [maintenanceForm, setMaintenanceForm] = useState({
    action: '',
    reviewed_by: '',
    notes: '',
  })

  const currentMachine = useMemo(
    () => machines.find((machine) => machine.machine_id === machineId),
    [machines, machineId],
  )

  const loadMachineDetails = useCallback(async (targetMachineId) => {
    if (!targetMachineId.trim()) return
    const encodedId = encodeURIComponent(targetMachineId.trim())
    const results = await Promise.allSettled([
      apiRequest(`/machines/${encodedId}/readings?limit=12`),
      apiRequest(`/machines/${encodedId}/analysis/latest`),
      apiRequest(`/maintenance/${encodedId}`),
    ])

    setHistory(results[0].status === 'fulfilled' ? results[0].value.readings : [])
    setAnalysis(results[1].status === 'fulfilled' ? results[1].value : null)
    setMaintenance(results[2].status === 'fulfilled' ? results[2].value : null)

    const unexpected = results.find(
      (result) => result.status === 'rejected' && result.reason.status !== 404,
    )
    if (unexpected) throw unexpected.reason
  }, [])

  const refreshDashboard = useCallback(async (targetMachineId) => {
    try {
      await apiRequest('/health')
      setApiStatus('online')
      setError('')
      const result = await apiRequest('/machines')
      setMachines(result.machines)
      const machineIsRegistered = result.machines.some(
        (machine) => machine.machine_id === targetMachineId.trim(),
      )
      if (machineIsRegistered) {
        await loadMachineDetails(targetMachineId)
      } else {
        setHistory([])
        setAnalysis(null)
        setMaintenance(null)
      }
    } catch (requestError) {
      setApiStatus('offline')
      setError(requestError.message || 'Could not load dashboard data.')
    } finally {
      setLoading(false)
    }
  }, [loadMachineDetails])

  useEffect(() => {
    void Promise.resolve().then(() => refreshDashboard('demo-machine-001'))
  }, [refreshDashboard])

  const updateReading = (event) => {
    const { name, value } = event.target
    setReading((current) => ({ ...current, [name]: value }))
  }

  const buildReadingPayload = () => ({
    machine_id: machineId.trim(),
    product_type: reading.product_type,
    air_temperature_k: Number(reading.air_temperature_k),
    process_temperature_k: Number(reading.process_temperature_k),
    rotational_speed_rpm: Number(reading.rotational_speed_rpm),
    torque_nm: Number(reading.torque_nm),
    tool_wear_min: Number(reading.tool_wear_min),
  })

  const saveReading = async (event) => {
    event.preventDefault()
    setSubmitting(true)
    setError('')
    setNotice('')
    try {
      const result = await apiRequest('/sensor-data', {
        method: 'POST',
        body: JSON.stringify(buildReadingPayload()),
      })
      setNotice(`Reading ${result.reading_id} saved for ${result.machine_id}.`)
      await refreshDashboard(result.machine_id)
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setSubmitting(false)
    }
  }

  const analyzeReading = async (event) => {
    event.preventDefault()
    setSubmitting(true)
    setError('')
    setNotice('')
    try {
      const result = await apiRequest('/analyze', {
        method: 'POST',
        body: JSON.stringify(buildReadingPayload()),
      })
      setAnalysis(result)
      setNotice(`Analysis ${result.analysis_id} saved for ${result.machine_id}.`)
      setActivePage('analysis')
      await refreshDashboard(result.machine_id)
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setSubmitting(false)
    }
  }

  const saveMaintenanceAction = async (event) => {
    event.preventDefault()
    setSubmitting(true)
    setError('')
    setNotice('')
    try {
      const result = await apiRequest(
        `/maintenance/${encodeURIComponent(machineId.trim())}/actions`,
        {
          method: 'POST',
          body: JSON.stringify(maintenanceForm),
        },
      )
      setNotice(`Maintenance action ${result.action_id} recorded.`)
      setMaintenanceForm({ action: '', reviewed_by: '', notes: '' })
      await refreshDashboard(machineId)
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setSubmitting(false)
    }
  }

  const pageTitle = NAV_ITEMS.find((item) => item.id === activePage)?.label ?? 'Overview'
  const latestResult = analysis ?? null
  const investigation = latestResult?.investigation

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">F<span>•</span></div>
          <div>
            <strong>FAILURE BEFORE<br />FAILURE</strong>
            <small>INDUSTRIAL INTELLIGENCE</small>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>
        <nav className="side-nav" aria-label="Main navigation">
          {NAV_ITEMS.map((item) => (
            <button
              type="button"
              key={item.id}
              className={`nav-item ${activePage === item.id ? 'active' : ''}`}
              aria-label={item.label}
              onClick={() => setActivePage(item.id)}
            >
              <IconMark>{item.icon}</IconMark>
              <span>{item.label}</span>
              {activePage === item.id && <i className="nav-indicator" />}
            </button>
          ))}
        </nav>

        <div className="sidebar-spacer" />
        <div className="sidebar-project">
          <span className="project-dot" />
          <div>
            <strong>Local prototype</strong>
            <small>AI4I 2020 · synthetic data</small>
          </div>
        </div>
        <div className="sidebar-foot">v1.0 · DEVELOPMENT</div>
      </aside>

      <main className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            <span>Workspace</span><b>/</b><strong>{pageTitle}</strong>
          </div>
          <div className="topbar-actions">
            <label className="machine-select">
              <span>Machine</span>
              <select
                aria-label="Selected machine"
                value={machineId}
                onChange={(event) => {
                  setMachineId(event.target.value)
                  setLoading(true)
                  refreshDashboard(event.target.value)
                }}
              >
                {!machines.some((machine) => machine.machine_id === machineId) && (
                  <option value={machineId}>{machineId}</option>
                )}
                {machines.map((machine) => (
                  <option key={machine.machine_id} value={machine.machine_id}>
                    {machine.machine_id}
                  </option>
                ))}
              </select>
            </label>
            <span className={`api-pill ${apiStatus}`}>
              <i />
              {apiStatus === 'online' ? 'API connected' : apiStatus === 'offline' ? 'API offline' : 'Connecting'}
            </span>
            <button
              type="button"
              className="icon-button"
              onClick={() => {
                setLoading(true)
                refreshDashboard(machineId)
              }}
              disabled={loading}
              aria-label="Refresh dashboard"
              title="Refresh dashboard"
            >
              ↻
            </button>
          </div>
        </header>

        <div className="content-wrap">
          <div className="page-heading">
            <div>
              <p className="eyebrow">INDUSTRIAL PREDICTIVE MAINTENANCE</p>
              <h1>{activePage === 'overview' ? 'Machine health overview' : pageTitle}</h1>
              <p className="page-subtitle">
                {activePage === 'overview'
                  ? 'Review readings, model signals, and maintenance activity.'
                  : `Working with machine ${machineId || '—'} · evidence for human review.`}
              </p>
            </div>
            <div className="heading-date">
              <span className="status-live"><i /> LOCAL SYSTEM</span>
              <span>{TODAY_LABEL}</span>
            </div>
          </div>

          {error && (
            <div className="alert alert-error" role="alert">
              <strong>Something needs attention</strong>
              <span>{error}</span>
              <button type="button" onClick={() => setError('')} aria-label="Dismiss error">×</button>
            </div>
          )}
          {notice && (
            <div className="alert alert-success" role="status">
              <strong>Saved</strong><span>{notice}</span>
              <button type="button" onClick={() => setNotice('')} aria-label="Dismiss message">×</button>
            </div>
          )}

          {activePage === 'overview' && (
            <OverviewPage
              machines={machines}
              currentMachine={currentMachine}
              machineId={machineId}
              latestResult={latestResult}
              history={history}
              loading={loading}
              goTo={setActivePage}
            />
          )}
          {activePage === 'sensors' && (
            <SensorsPage
              machineId={machineId}
              setMachineId={setMachineId}
              reading={reading}
              updateReading={updateReading}
              saveReading={saveReading}
              analyzeReading={analyzeReading}
              submitting={submitting}
              history={history}
            />
          )}
          {activePage === 'analysis' && (
            <AnalysisPage
              analysis={latestResult}
              loading={loading}
              goTo={setActivePage}
            />
          )}
          {activePage === 'investigation' && (
            <InvestigationPage investigation={investigation} analysis={latestResult} />
          )}
          {activePage === 'maintenance' && (
            <MaintenancePage
              maintenance={maintenance}
              form={maintenanceForm}
              setForm={setMaintenanceForm}
              onSubmit={saveMaintenanceAction}
              submitting={submitting}
              machineId={machineId}
            />
          )}

          <footer className="disclaimer">
            <span>Prototype notice</span>
            <p>
              AI4I is synthetic research data. An anomaly is not proof of failure.
              This project is not a certified industrial safety system.
            </p>
            <a href="https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset" target="_blank" rel="noreferrer">
              Dataset source ↗
            </a>
          </footer>
        </div>
      </main>
    </div>
  )
}

function OverviewPage({ machines, currentMachine, machineId, latestResult, history, loading, goTo }) {
  const anomalyFlagged = latestResult?.anomaly?.flagged_as_anomaly
  const predictedFailure = latestResult?.classifier?.predicted_label === 1

  return (
    <>
      <section className="overview-banner">
        <div className="banner-copy">
          <span className="banner-kicker">ACTIVE MACHINE</span>
          <h2>{currentMachine?.machine_id ?? machineId}</h2>
          <p>
            {currentMachine
              ? `Monitoring ${currentMachine.reading_count} saved sensor readings.`
              : 'No readings saved for this machine yet. Submit a sensor reading to begin.'}
          </p>
        </div>
        <div className="banner-metric">
          <span>LAST ANALYSIS</span>
          <strong>{latestResult?.analysis_id ? `#${latestResult.analysis_id}` : '—'}</strong>
          <small>{latestResult ? 'Saved locally' : 'Awaiting first run'}</small>
        </div>
        <button type="button" className="button button-light" onClick={() => goTo('sensors')}>
          Analyze a reading <span aria-hidden="true">→</span>
        </button>
        <div className="banner-orbit orbit-one" />
        <div className="banner-orbit orbit-two" />
      </section>

      <section className="stat-grid" aria-label="System summary">
        <StatCard label="Registered machines" value={machines.length} icon="⌂" detail="Tracked locally" />
        <StatCard label="Sensor readings" value={machines.reduce((sum, item) => sum + item.reading_count, 0)} icon="⌁" detail="Stored in SQLite" />
        <StatCard
          label="Classifier signal"
          value={latestResult ? (predictedFailure ? 'Review' : 'No flag') : 'No data'}
          icon="◎"
          detail={latestResult ? `${formatNumber(latestResult.classifier.model_estimated_failure_probability * 100, 1)}% model estimate` : 'Run an analysis to view'}
          tone={predictedFailure ? 'amber' : ''}
        />
        <StatCard
          label="Anomaly signal"
          value={latestResult ? (anomalyFlagged ? 'Unusual' : 'Within threshold') : 'No data'}
          icon="⌖"
          detail={latestResult ? `Score ${formatNumber(latestResult.anomaly.reconstruction_error, 3)}` : 'Run an analysis to view'}
          tone={anomalyFlagged ? 'amber' : ''}
        />
      </section>

      <div className="dashboard-grid">
        <section className="panel trend-panel">
          <PanelHeading
            eyebrow="SENSOR MONITORING"
            title="Temperature trend"
            action={<button type="button" className="text-button" onClick={() => goTo('sensors')}>View readings →</button>}
          />
          <TrendChart readings={history} />
          <div className="trend-foot">
            <span><i className="legend-dot air" /> Air · K</span>
            <span><i className="legend-dot process" /> Process · K</span>
            <span>{history.length} latest readings</span>
          </div>
        </section>

        <section className="panel signal-panel">
          <PanelHeading eyebrow="LATEST MODEL OUTPUT" title="Signals to review" />
          {latestResult ? (
            <div className="signal-list">
              <SignalRow
                icon="◎"
                label="Failure classifier"
                value={latestResult.classifier.predicted_label === 1 ? 'Positive label' : 'Negative label'}
                detail={`${formatNumber(latestResult.classifier.model_estimated_failure_probability * 100, 1)}% estimated probability`}
                tone={latestResult.classifier.predicted_label === 1 ? 'warning' : 'good'}
              />
              <SignalRow
                icon="⌖"
                label="Autoencoder"
                value={latestResult.anomaly.flagged_as_anomaly ? 'Above threshold' : 'Below threshold'}
                detail={`Error ${formatNumber(latestResult.anomaly.reconstruction_error, 4)} · limit ${formatNumber(latestResult.anomaly.threshold, 4)}`}
                tone={latestResult.anomaly.flagged_as_anomaly ? 'warning' : 'good'}
              />
              <div className="signal-note">
                <IconMark>i</IconMark>
                Model signals are evidence for review, not diagnoses or guarantees.
              </div>
            </div>
          ) : (
            <EmptyState title="No analysis yet">
              Submit a reading to see the classifier and anomaly signals here.
            </EmptyState>
          )}
        </section>

        <section className="panel history-panel">
          <PanelHeading
            eyebrow="RECENT ACTIVITY"
            title="Latest sensor readings"
            action={<button type="button" className="text-button" onClick={() => goTo('sensors')}>Open sensor data →</button>}
          />
          <ReadingsTable readings={history.slice(0, 5)} loading={loading} />
        </section>

        <section className="panel action-panel">
          <PanelHeading eyebrow="NEXT STEP" title="Human review" />
          <div className="review-card">
            <div className="review-icon"><IconMark>✓</IconMark></div>
            <div>
              <strong>Keep a maintenance record</strong>
              <p>Review suggested checks and log the action taken by your team.</p>
            </div>
          </div>
          <button type="button" className="button button-primary full-button" onClick={() => goTo('maintenance')}>
            Open maintenance review <span aria-hidden="true">→</span>
          </button>
        </section>
      </div>
    </>
  )
}

function SensorsPage({ machineId, setMachineId, reading, updateReading, saveReading, analyzeReading, submitting, history }) {
  return (
    <div className="page-grid">
      <section className="panel form-panel">
        <PanelHeading eyebrow="INPUT" title="Industrial sensor reading" />
        <p className="panel-intro">
          Enter a reading to store it or run the available models. Analysis also saves this reading.
        </p>
        <form className="reading-form" onSubmit={analyzeReading}>
          <label className="field field-wide">
            <span>Machine ID</span>
            <input
              name="machine_id"
              value={machineId}
              onChange={(event) => setMachineId(event.target.value)}
              maxLength={100}
              required
            />
            <small>New IDs are registered automatically when you save or analyze.</small>
          </label>
          <label className="field">
            <span>Product type</span>
            <select name="product_type" value={reading.product_type} onChange={updateReading}>
              <option value="L">L · Low</option>
              <option value="M">M · Medium</option>
              <option value="H">H · High</option>
            </select>
          </label>
          <label className="field">
            <span>Air temperature <i>K</i></span>
            <input name="air_temperature_k" type="number" min="0.1" step="0.1" value={reading.air_temperature_k} onChange={updateReading} required />
          </label>
          <label className="field">
            <span>Process temperature <i>K</i></span>
            <input name="process_temperature_k" type="number" min="0.1" step="0.1" value={reading.process_temperature_k} onChange={updateReading} required />
          </label>
          <label className="field">
            <span>Rotational speed <i>rpm</i></span>
            <input name="rotational_speed_rpm" type="number" min="0" step="1" value={reading.rotational_speed_rpm} onChange={updateReading} required />
          </label>
          <label className="field">
            <span>Torque <i>Nm</i></span>
            <input name="torque_nm" type="number" min="0" step="0.1" value={reading.torque_nm} onChange={updateReading} required />
          </label>
          <label className="field">
            <span>Tool wear <i>min</i></span>
            <input name="tool_wear_min" type="number" min="0" step="1" value={reading.tool_wear_min} onChange={updateReading} required />
          </label>
          <div className="form-actions field-wide">
            <button className="button button-primary" type="submit" disabled={submitting || !machineId.trim()}>
              {submitting ? 'Analyzing…' : 'Analyze & save reading'}
              <span aria-hidden="true">→</span>
            </button>
            <button className="button button-outline" type="button" disabled={submitting || !machineId.trim()} onClick={saveReading}>
              Save reading only
            </button>
          </div>
        </form>
        <div className="form-footnote">
          <IconMark>i</IconMark>
          Use demonstration or dataset values. Do not treat this prototype as a live machine safety monitor.
        </div>
      </section>

      <section className="panel readings-panel">
        <PanelHeading eyebrow="HISTORY" title={`Recent readings · ${machineId}`} />
        <ReadingsTable readings={history} />
      </section>
    </div>
  )
}

function AnalysisPage({ analysis, loading, goTo }) {
  if (!analysis) {
    return (
      <section className="panel empty-panel">
        <EmptyState title={loading ? 'Loading analysis…' : 'No saved analysis'}>
          Analyze a sensor reading to see model results for this machine.
        </EmptyState>
        <button type="button" className="button button-primary" onClick={() => goTo('sensors')}>Enter a sensor reading →</button>
      </section>
    )
  }

  const resultIsPositive = analysis.classifier.predicted_label === 1
  return (
    <>
      <div className="result-banner">
        <div className={`result-mark ${resultIsPositive || analysis.anomaly.flagged_as_anomaly ? 'warning' : ''}`}>
          {resultIsPositive || analysis.anomaly.flagged_as_anomaly ? '!' : '✓'}
        </div>
        <div>
          <span className="eyebrow">ANALYSIS #{analysis.analysis_id ?? '—'} · {analysis.machine_id}</span>
          <h2>{resultIsPositive || analysis.anomaly.flagged_as_anomaly ? 'Signals need human review' : 'No model threshold was exceeded'}</h2>
          <p>Model output is an estimate based on synthetic AI4I training data.</p>
        </div>
        <button type="button" className="button button-outline" onClick={() => goTo('investigation')}>View investigation →</button>
      </div>
      <section className="analysis-cards">
        <div className="panel model-card">
          <div className="model-card-head"><span className="model-icon classifier">◎</span><span className="model-tag">SUPERVISED MODEL</span></div>
          <h3>Failure classifier</h3>
          <div className="metric-large">{formatNumber(analysis.classifier.model_estimated_failure_probability * 100, 1)}<small>%</small></div>
          <p>Model-estimated probability for the known failure label.</p>
          <div className="result-detail">
            <span>Predicted label</span>
            <strong>{analysis.classifier.predicted_label === 1 ? 'Failure label' : 'No failure label'}</strong>
          </div>
          <div className="result-detail">
            <span>Decision threshold</span><strong>{formatNumber(analysis.classifier.decision_threshold * 100, 0)}%</strong>
          </div>
          <p className="model-note">{analysis.classifier.note}</p>
        </div>
        <div className="panel model-card">
          <div className="model-card-head"><span className="model-icon anomaly">⌖</span><span className="model-tag">UNSUPERVISED MODEL</span></div>
          <h3>Autoencoder anomaly</h3>
          <div className="metric-large">{formatNumber(analysis.anomaly.reconstruction_error, 4)}</div>
          <p>Reconstruction error compared with the validation threshold.</p>
          <div className="result-detail">
            <span>Validation threshold</span><strong>{formatNumber(analysis.anomaly.threshold, 4)}</strong>
          </div>
          <div className="result-detail">
            <span>Threshold status</span><strong>{analysis.anomaly.flagged_as_anomaly ? 'Above threshold' : 'Below threshold'}</strong>
          </div>
          <p className="model-note">{analysis.anomaly.note}</p>
        </div>
      </section>
      <div className="panel model-limit">
        <IconMark>i</IconMark>
        <p><strong>Interpret carefully.</strong> A classifier label or unusual sensor pattern does not prove a machine will fail. Confirm findings with equipment-specific operating limits and a qualified maintenance review.</p>
      </div>
    </>
  )
}

function InvestigationPage({ investigation, analysis }) {
  if (!investigation) {
    return (
      <section className="panel empty-panel">
        <EmptyState title="No investigation report yet">
          Run an analysis to generate rule-based evidence and suggested checks.
        </EmptyState>
      </section>
    )
  }

  return (
    <div className="investigation-layout">
      <section className="panel report-summary">
        <span className="eyebrow">EVIDENCE REPORT · ANALYSIS #{analysis?.analysis_id ?? '—'}</span>
        <h2>Investigation findings</h2>
        <p>{investigation.summary}</p>
        <div className="report-meta">
          <span>Machine <strong>{analysis?.machine_id ?? '—'}</strong></span>
          <span>Evidence items <strong>{investigation.evidence.length}</strong></span>
        </div>
      </section>
      <section className="panel evidence-panel">
        <PanelHeading eyebrow="OBSERVATIONS" title="What the system found" />
        {investigation.evidence.length ? (
          <div className="evidence-list">
            {investigation.evidence.map((item, index) => (
              <article className="evidence-item" key={`${item.rule_id}-${index}`}>
                <div className="evidence-index">{String(index + 1).padStart(2, '0')}</div>
                <div>
                  <span className="evidence-rule">{item.rule_id}</span>
                  <p className="evidence-observation">{item.observation}</p>
                  <p className="evidence-meaning">{item.interpretation}</p>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <EmptyState title="No configured patterns were observed">
            This does not establish that the machine is failure-free.
          </EmptyState>
        )}
      </section>
      <section className="panel checks-panel">
        <PanelHeading eyebrow="HUMAN REVIEW" title="Suggested checks" />
        {investigation.recommended_checks.length ? (
          <ol className="check-list">
            {investigation.recommended_checks.map((check, index) => (
              <li key={`${check}-${index}`}><span>{String(index + 1).padStart(2, '0')}</span>{check}</li>
            ))}
          </ol>
        ) : <p className="muted-copy">No specific checks were suggested for this result.</p>}
      </section>
      <section className="panel limitations-panel">
        <PanelHeading eyebrow="LIMITATIONS" title="Read before acting" />
        <ul>
          {investigation.limitations.map((item) => <li key={item}>{item}</li>)}
        </ul>
      </section>
    </div>
  )
}

function MaintenancePage({ maintenance, form, setForm, onSubmit, submitting, machineId }) {
  const actions = maintenance?.actions ?? []
  const suggestedChecks = maintenance?.suggested_checks ?? []

  return (
    <div className="maintenance-grid">
      <section className="panel checks-panel">
        <PanelHeading eyebrow="FROM LATEST INVESTIGATION" title="Suggested maintenance checks" />
        {suggestedChecks.length ? (
          <ol className="check-list">
            {suggestedChecks.map((check, index) => (
              <li key={`${check}-${index}`}><span>{String(index + 1).padStart(2, '0')}</span>{check}</li>
            ))}
          </ol>
        ) : (
          <EmptyState title="No suggested checks saved">
            Analyze a reading for this machine to create evidence-based review suggestions.
          </EmptyState>
        )}
      </section>
      <section className="panel maintenance-form-panel">
        <PanelHeading eyebrow="HUMAN-IN-THE-LOOP" title="Record a maintenance action" />
        <p className="panel-intro">Log what a person reviewed or did. These notes are stored locally against {machineId}.</p>
        <form className="maintenance-form" onSubmit={onSubmit}>
          <label className="field">
            <span>Action taken</span>
            <input
              value={form.action}
              onChange={(event) => setForm((current) => ({ ...current, action: event.target.value }))}
              maxLength={1000}
              placeholder="e.g. Inspected cooling airflow"
              required
            />
          </label>
          <label className="field">
            <span>Reviewed by</span>
            <input
              value={form.reviewed_by}
              onChange={(event) => setForm((current) => ({ ...current, reviewed_by: event.target.value }))}
              maxLength={100}
              placeholder="Technician name"
              required
            />
          </label>
          <label className="field">
            <span>Notes <i>optional</i></span>
            <textarea
              value={form.notes}
              onChange={(event) => setForm((current) => ({ ...current, notes: event.target.value }))}
              maxLength={2000}
              rows={4}
              placeholder="Add the human review outcome or follow-up."
            />
          </label>
          <button type="submit" className="button button-primary" disabled={submitting || !maintenance}>
            {submitting ? 'Saving…' : 'Save review action'} <span aria-hidden="true">→</span>
          </button>
          {!maintenance && <small className="form-hint">Register this machine by saving or analyzing a sensor reading first.</small>}
        </form>
      </section>
      <section className="panel action-history-panel">
        <PanelHeading eyebrow="AUDIT TRAIL" title="Recorded actions" />
        {actions.length ? (
          <div className="action-list">
            {actions.map((action) => (
              <article className="action-item" key={action.action_id}>
                <span className="action-check">✓</span>
                <div>
                  <strong>{action.action}</strong>
                  <p>{action.notes || 'No additional notes.'}</p>
                  <small>{action.reviewed_by} · {formatDate(action.created_at)}</small>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <EmptyState title="No review actions recorded">
            Actions entered by a human reviewer will appear here.
          </EmptyState>
        )}
      </section>
    </div>
  )
}

function StatCard({ label, value, icon, detail, tone = '' }) {
  return (
    <article className="stat-card">
      <div className="stat-top"><span>{label}</span><IconMark className={tone}>{icon}</IconMark></div>
      <strong className={`stat-value ${tone}`}>{value}</strong>
      <small>{detail}</small>
    </article>
  )
}

function PanelHeading({ eyebrow, title, action }) {
  return (
    <div className="panel-heading">
      <div><span className="eyebrow">{eyebrow}</span><h2>{title}</h2></div>
      {action}
    </div>
  )
}

function SignalRow({ icon, label, value, detail, tone }) {
  return (
    <div className="signal-row">
      <div className={`signal-icon ${tone}`}><IconMark>{icon}</IconMark></div>
      <div className="signal-copy"><strong>{label}</strong><small>{detail}</small></div>
      <span className={`signal-state ${tone}`}>{value}</span>
    </div>
  )
}

function ReadingsTable({ readings, loading = false }) {
  if (loading && readings.length === 0) {
    return <EmptyState title="Loading readings">Checking local sensor history.</EmptyState>
  }
  if (!readings.length) {
    return <EmptyState title="No readings available">Saved or analyzed readings will appear here.</EmptyState>
  }

  return (
    <div className="table-scroll">
      <table className="readings-table">
        <thead>
          <tr>
            <th>Recorded</th><th>Type</th><th>Air · K</th><th>Process · K</th><th>Speed · rpm</th><th>Torque · Nm</th><th>Wear · min</th>
          </tr>
        </thead>
        <tbody>
          {readings.map((item) => (
            <tr key={item.reading_id}>
              <td>{formatDate(item.recorded_at)}</td>
              <td><span className="type-pill">{item.product_type}</span></td>
              <td>{formatNumber(item.air_temperature_k)}</td>
              <td>{formatNumber(item.process_temperature_k)}</td>
              <td>{formatNumber(item.rotational_speed_rpm, 0)}</td>
              <td>{formatNumber(item.torque_nm)}</td>
              <td>{formatNumber(item.tool_wear_min, 0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default App
