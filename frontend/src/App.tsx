import './App.css'

function App() {
  return (
    <main className="chat">
      <header className="chat-header">
        <span className="brand">Finance Assistant</span>
        <span className="document-badge">Coca-Cola · 2022</span>
      </header>

      <section className="welcome">
        <span className="welcome-label">YOUR FINANCIAL RESEARCH ASSISTANT</span>
        <h1>Make sense of the numbers.</h1>
        <p>
          Ask about Coca-Cola’s annual report. Get answers with document and
          page references.
        </p>
      </section>

      <footer className="chat-footer">
        <div className="composer">
          <button
            type="button"
            className="attach-button"
            aria-label="Attach files (coming soon)"
            title="File attachments — coming soon"
            disabled
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 5v14M5 12h14" />
            </svg>
          </button>

          <textarea
            aria-label="Your question about the report"
            placeholder="What was Coca-Cola's revenue in 2022?"
            rows={1}
          />

          <button
            type="button"
            className="send-button"
            aria-label="Send question"
            disabled
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 19V5m-6 6 6-6 6 6" />
            </svg>
          </button>
        </div>

        <p className="footer-note">
          Answers may contain errors. Check the sources.
        </p>
      </footer>
    </main>
  )
}

export default App
