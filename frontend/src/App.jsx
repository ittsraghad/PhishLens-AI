
import { useState } from 'react'
import './App.css'

const API_BASE = (
  import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
).replace(/\/+$/, '')

function ShieldIcon({ size = 24 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11Z" />
      <path d="m9 12 2 2 4-4" />
    </svg>
  )
}

function SearchIcon({ size = 21 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="11" cy="11" r="7" />
      <path d="m16 16 4 4" />
      <path d="M11 8v6m-3-3h6" />
    </svg>
  )
}

function TextIcon({ size = 23 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M5 5h14" />
      <path d="M12 5v14" />
      <path d="M8 19h8" />
      <path d="M8 9h8" />
    </svg>
  )
}

function FeaturesIcon({ size = 23 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <rect x="4" y="4" width="6" height="6" rx="1.5" />
      <rect x="14" y="4" width="6" height="6" rx="1.5" />
      <rect x="4" y="14" width="6" height="6" rx="1.5" />
      <rect x="14" y="14" width="6" height="6" rx="1.5" />
    </svg>
  )
}

function ModelIcon({ size = 23 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="m12 3 2.2 5.8L20 11l-5.8 2.2L12 19l-2.2-5.8L4 11l5.8-2.2L12 3Z" />
      <path d="m19 15 .9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9L19 15Z" />
    </svg>
  )
}

function App() {
  const [emailText, setEmailText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const examples = {
    suspicious:
      'Dear customer, your account has been suspended. Click here to verify your password immediately.',
    safe:
      'Hi Sarah, our team meeting is scheduled for tomorrow at 10 AM. Please review the project notes before the meeting. Thanks.',
  }

  async function analyzeEmail() {
    if (!emailText.trim()) {
      setError('Please enter an email message before analyzing.')
      setResult(null)
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify({
          body: emailText,
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(
          typeof data.detail === 'string'
            ? data.detail
            : 'Unable to analyze this email. Please try again.',
        )
      }

      setResult(data)
    } catch (err) {
      if (err.message === 'Failed to fetch') {
        setError(
          'Unable to connect to the analysis service. Please try again later.',
        )
      } else {
        setError(err.message || 'Something went wrong. Please try again.')
      }
    } finally {
      setLoading(false)
    }
  }

  function loadExample(type) {
    setEmailText(examples[type])
    setResult(null)
    setError('')
  }

  function handleTextChange(event) {
    setEmailText(event.target.value)
    setResult(null)
    setError('')
  }

  const isSuspicious = result?.classification === 'suspicious'

  const score = result
    ? (Number(result.suspicious_class_probability) * 100).toFixed(2)
    : null

  const scoreWidth = result
    ? Math.min(
      100,
      Math.max(
        0,
        Number(result.suspicious_class_probability) * 100,
      ),
    )
    : 0

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#home" aria-label="PhishLens AI home">
          <span className="brand-icon">
            <ShieldIcon size={25} />
          </span>

          <span className="brand-text">
            <strong>
              PhishLens <span>AI</span>
            </strong>
            <small>EMAIL INTELLIGENCE</small>
          </span>
        </a>

        <nav className="header-nav" aria-label="Main navigation">
          <a href="#analyzer">Email analyzer</a>
          <a href="#how-it-works">How it works</a>
          <a href="#about">About</a>
        </nav>

        <a className="header-button" href="#analyzer">
          Analyze an email
        </a>
      </header>

      <main id="home">
        <section className="hero">
          <div className="hero-badge">
            <span className="badge-sparkle">✦</span>
            MACHINE LEARNING · EMAIL ANALYSIS
          </div>

          <h1>
            A clearer view of
            <br />
            <span>what’s in your inbox.</span>
          </h1>

          <p className="hero-description">
            Analyze email text with machine learning and identify messages
            that may deserve a closer look.
          </p>

          <a className="hero-link" href="#analyzer">
            Try the email analyzer <span>↓</span>
          </a>
        </section>

        <section className="analyzer-section" id="analyzer">
          <div className="section-heading">
            <span className="section-eyebrow">EMAIL ANALYZER</span>
            <h2>Let’s inspect a message.</h2>
            <p>
              Paste the email text below to see your model’s prediction.
            </p>
          </div>

          <div className="analyzer-card">
            <div className="input-section">
              <div className="field-heading">
                <label htmlFor="email-input">Email content</label>
                <span>{emailText.length.toLocaleString()} / 100,000</span>
              </div>

              <textarea
                id="email-input"
                value={emailText}
                onChange={handleTextChange}
                placeholder="Paste the email message here..."
                maxLength={100000}
                spellCheck={false}
              />

              <div className="example-section">
                <span className="example-label">Need an example?</span>

                <div className="example-buttons">
                  <button
                    type="button"
                    className="example-button"
                    onClick={() => loadExample('suspicious')}
                    disabled={loading}
                  >
                    Suspicious email
                  </button>

                  <button
                    type="button"
                    className="example-button"
                    onClick={() => loadExample('safe')}
                    disabled={loading}
                  >
                    Routine email
                  </button>
                </div>
              </div>

              {error && (
                <div className="error-message" role="alert">
                  {error}
                </div>
              )}

              <button
                type="button"
                className="analyze-button"
                onClick={analyzeEmail}
                disabled={loading}
              >
                {loading ? (
                  <>
                    <span className="spinner" />
                    Analyzing message...
                  </>
                ) : (
                  <>
                    <SearchIcon size={20} />
                    Analyze email
                  </>
                )}
              </button>

              <p className="privacy-note">
                <ShieldIcon size={16} />
                <span>
                  For privacy, avoid submitting passwords, personal details,
                  or confidential email content.
                </span>
              </p>
            </div>

            <div className="result-section" aria-live="polite">
              {!result && !loading && (
                <div className="result-placeholder">
                  <div className="placeholder-icon">
                    <SearchIcon size={30} />
                  </div>

                  <h3>Your results will appear here.</h3>

                  <p>
                    Submit an email to view the classification produced by
                    the machine learning model.
                  </p>

                  <span className="placeholder-label">
                    AWAITING ANALYSIS
                  </span>
                </div>
              )}

              {loading && (
                <div className="result-placeholder">
                  <div className="placeholder-icon loading-icon">
                    <span className="spinner" />
                  </div>

                  <h3>Analyzing your message</h3>

                  <p>
                    The text is being processed by the machine learning
                    pipeline.
                  </p>
                </div>
              )}

              {result && (
                <div
                  className={`result-content ${isSuspicious ? 'result-suspicious' : 'result-safe'
                    }`}
                >
                  <span className="result-eyebrow">
                    MODEL CLASSIFICATION
                  </span>

                  <div className="result-status">
                    <span className="result-status-icon">
                      {isSuspicious ? '!' : '✓'}
                    </span>

                    <div>
                      <h3>
                        {isSuspicious
                          ? 'Suspicious'
                          : 'Classified as safe'}
                      </h3>

                      <p>
                        {isSuspicious
                          ? 'The model placed this message in the suspicious class.'
                          : 'The model placed this message in the safe class.'}
                      </p>
                    </div>
                  </div>

                  <div className="score-card">
                    <div className="score-heading">
                      <span>Suspicious-class score</span>
                      <strong>{score}%</strong>
                    </div>

                    <div
                      className="score-track"
                      role="progressbar"
                      aria-label="Suspicious-class score"
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={Number(score)}
                    >
                      <div
                        className="score-fill"
                        style={{ width: `${scoreWidth}%` }}
                      />
                    </div>

                    <p>
                      This is the model’s estimated score for the suspicious
                      class, not a verified measure of real-world risk.
                    </p>
                  </div>

                  <div className="result-message">
                    <strong>What this means</strong>
                    <p>{result.message}</p>
                  </div>

                  <div className="prediction-row">
                    <span>Model prediction</span>
                    <strong>{result.prediction}</strong>
                  </div>

                  <p className="result-disclaimer">
                    {result.disclaimer}
                  </p>

                  <button
                    type="button"
                    className="reset-button"
                    onClick={() => {
                      setResult(null)
                      setError('')
                      setEmailText('')
                    }}
                  >
                    Analyze another email
                  </button>
                </div>
              )}
            </div>
          </div>

          <p className="demo-note">
            PhishLens AI is an educational machine learning project.
            Its predictions may be incorrect and should not be treated as
            proof that an email is safe or malicious.
          </p>
        </section>

        <section className="how-section" id="how-it-works">
          <div className="section-heading">
            <span className="section-eyebrow">HOW IT WORKS</span>
            <h2>From text to prediction.</h2>
            <p>
              A simple machine learning pipeline powers the analysis.
            </p>
          </div>

          <div className="steps-grid">
            <article className="step-card">
              <span className="step-number">01</span>

              <div className="step-icon">
                <TextIcon />
              </div>

              <h3>Text input</h3>
              <p>
                The email text is provided to the analysis pipeline.
              </p>
            </article>

            <article className="step-card">
              <span className="step-number">02</span>

              <div className="step-icon">
                <FeaturesIcon />
              </div>

              <h3>TF-IDF features</h3>
              <p>
                Text is transformed into numerical features for the model.
              </p>
            </article>

            <article className="step-card">
              <span className="step-number">03</span>

              <div className="step-icon">
                <ModelIcon />
              </div>

              <h3>ML prediction</h3>
              <p>
                Logistic Regression predicts a message category.
              </p>
            </article>
          </div>
        </section>

        <section className="about-section" id="about">
          <div className="about-icon">
            <ShieldIcon size={25} />
          </div>

          <div>
            <h2>About PhishLens AI</h2>
            <p>
              An educational project exploring how machine learning can
              help classify email messages as safe or suspicious.
            </p>
          </div>
        </section>
      </main>

      <footer className="site-footer">
        <a className="footer-brand" href="#home">
          <ShieldIcon size={18} />
          PhishLens AI
        </a>

        <span>Machine learning for email analysis</span>
      </footer>
    </div>
  )
}

export default App
