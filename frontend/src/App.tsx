import { FormEvent, startTransition, useDeferredValue, useEffect, useState } from "react";

type MetadataValue = string | number | boolean | null;

type SourceItem = {
  source_id: string;
  title: string;
  snippet: string;
  source_type: string;
  url: string | null;
  metadata: Record<string, MetadataValue>;
};

type EvaluationScore = {
  metric: string;
  score: number;
  rationale: string;
};

type ClaimEvidence = {
  source_id: string;
  quote: string;
};

type AnswerClaim = {
  claim_text: string;
  supporting_source_ids: string[];
  supporting_quotes: ClaimEvidence[];
  confidence: "low" | "medium" | "high";
  limitations: string[];
  conflicts: string[];
  support_status: "supported" | "unsupported" | "conflicting";
};

type CorpusStats = {
  source_document_count: number;
  chunk_count: number;
  corpus_version_id: string;
};

type AnswerMode = "concise" | "detailed" | "evidence_table";

type ResearchResponse = {
  run_id: string;
  corpus_version_id: string;
  corpus_stats: CorpusStats;
  question: string;
  answer: string;
  claims: AnswerClaim[];
  sources: SourceItem[];
  evaluation: EvaluationScore[];
  execution_trace: string[];
};

type RunSummary = {
  run_id: string;
  question: string;
  answer_preview: string;
  answer_mode: AnswerMode;
  source_count: number;
  created_at: string;
  latency_ms: number | null;
};

type WorkspaceRun = ResearchResponse & {
  answer_mode: AnswerMode;
  created_at: string;
  latency_ms: number | null;
};

type RunSourceDetail = {
  run_id: string;
  source_id: string;
  title: string;
  source_type: string;
  snippet: string;
  text: string;
  source_document_id: string | null;
  chunk_id: string | null;
  page_start: number | null;
  page_end: number | null;
  section_title: string | null;
  highlights: string[];
  metadata: Record<string, MetadataValue>;
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

const QUICK_PROMPTS = [
  "What are the main challenges of RAG according to the internal corpus?",
  "What is the difference between LangChain and LangGraph in the indexed documents?",
  "How many indexed chunks are stored in the project corpus?",
];

const ANSWER_MODES: { value: AnswerMode; label: string }[] = [
  { value: "detailed", label: "Detailed" },
  { value: "concise", label: "Concise" },
  { value: "evidence_table", label: "Evidence table" },
];

function App() {
  const [question, setQuestion] = useState(QUICK_PROMPTS[0]);
  const [topK, setTopK] = useState(4);
  const [answerMode, setAnswerMode] = useState<AnswerMode>("detailed");
  const [result, setResult] = useState<WorkspaceRun | ResearchResponse | null>(null);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedSource, setSelectedSource] = useState<RunSourceDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [sourceLoading, setSourceLoading] = useState(false);
  const [feedbackStatus, setFeedbackStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const deferredQuestion = useDeferredValue(question);

  useEffect(() => {
    void refreshHistory();
  }, []);

  const loweredQuestion = deferredQuestion.toLowerCase();
  let questionLens = "General research";
  if (/(how many|quantos|count|total|sum)/.test(loweredQuestion)) {
    questionLens = "Quantitative query";
  } else if (/(compar|differenc|diferenc|vs)/.test(loweredQuestion)) {
    questionLens = "Comparative question";
  } else if (/(challenge|desafio|risk|limita)/.test(loweredQuestion)) {
    questionLens = "Risk analysis";
  }

  async function refreshHistory() {
    setHistoryLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/runs?limit=12`);
      if (response.ok) {
        setRuns((await response.json()) as RunSummary[]);
      }
    } catch {
      setRuns([]);
    } finally {
      setHistoryLoading(false);
    }
  }

  async function loadRun(runId: string) {
    setError(null);
    setSelectedSource(null);
    try {
      const response = await fetch(`${API_BASE_URL}/runs/${runId}`);
      if (!response.ok) {
        setError(`Run lookup failed (${response.status})`);
        return;
      }
      const payload = (await response.json()) as WorkspaceRun;
      startTransition(() => {
        setResult(payload);
      });
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Run lookup failed.");
    }
  }

  async function loadSource(source: SourceItem) {
    if (!result) {
      return;
    }
    setSourceLoading(true);
    setSelectedSource(null);
    try {
      const response = await fetch(
        `${API_BASE_URL}/runs/${result.run_id}/sources/${encodeURIComponent(source.source_id)}`,
      );
      if (response.ok) {
        setSelectedSource((await response.json()) as RunSourceDetail);
      } else {
        setSelectedSource(sourceToDetail(result.run_id, source));
      }
    } catch {
      setSelectedSource(sourceToDetail(result.run_id, source));
    } finally {
      setSourceLoading(false);
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!question.trim()) {
      setError("Enter a question to start the research workflow.");
      return;
    }

    setLoading(true);
    setError(null);
    setFeedbackStatus(null);
    setSelectedSource(null);

    try {
      const response = await fetch(`${API_BASE_URL}/research`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question,
          top_k: topK,
          answer_mode: answerMode,
        }),
      });

      if (!response.ok) {
        throw new Error(`Research request failed (${response.status})`);
      }

      const payload = (await response.json()) as ResearchResponse;
      startTransition(() => {
        setResult(payload);
      });
      await refreshHistory();
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Unexpected failure while calling the API.",
      );
    } finally {
      setLoading(false);
    }
  }

  async function submitFeedback(rating: "up" | "down") {
    if (!result) {
      return;
    }
    setFeedbackStatus(null);
    try {
      const response = await fetch(`${API_BASE_URL}/runs/${result.run_id}/feedback`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          rating,
          comment: rating === "up" ? "Useful answer." : "Needs review.",
          add_to_eval: rating === "down",
          corrected_answer: null,
        }),
      });
      setFeedbackStatus(response.ok ? "Feedback recorded" : `Feedback failed (${response.status})`);
    } catch (requestError) {
      setFeedbackStatus(requestError instanceof Error ? requestError.message : "Feedback failed.");
    }
  }

  return (
    <main className="shell">
      <section className="workspace">
        <aside className="panel panel-form">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Composer</p>
              <h2>Research run</h2>
            </div>
            <div className="lens-pill">{questionLens}</div>
          </div>

          <form className="research-form" onSubmit={handleSubmit}>
            <label className="field">
              <span>Question</span>
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="What does the internal corpus say about retrieval quality?"
                rows={7}
              />
            </label>

            <div className="mode-control" aria-label="Answer mode">
              {ANSWER_MODES.map((mode) => (
                <button
                  key={mode.value}
                  type="button"
                  className={answerMode === mode.value ? "mode-button active" : "mode-button"}
                  onClick={() => setAnswerMode(mode.value)}
                >
                  {mode.label}
                </button>
              ))}
            </div>

            <div className="controls">
              <label className="field compact">
                <span>Top K</span>
                <input
                  type="range"
                  min={1}
                  max={10}
                  value={topK}
                  onChange={(event) => setTopK(Number(event.target.value))}
                />
                <strong>{topK} evidence items</strong>
              </label>
              <button className="submit-button" type="submit" disabled={loading}>
                {loading ? "Researching..." : "Run research"}
              </button>
            </div>
          </form>

          <div className="prompt-bank">
            {QUICK_PROMPTS.map((prompt) => (
              <button
                key={prompt}
                type="button"
                className="prompt-chip"
                onClick={() => setQuestion(prompt)}
              >
                {prompt}
              </button>
            ))}
          </div>

          <section className="history-section">
            <div className="section-heading">
              <p className="section-label">Run history</p>
              <span>{historyLoading ? "loading" : `${runs.length} runs`}</span>
            </div>
            <div className="history-list">
              {runs.map((run) => (
                <button
                  key={run.run_id}
                  type="button"
                  className="history-item"
                  onClick={() => void loadRun(run.run_id)}
                >
                  <strong>{run.question}</strong>
                  <span>
                    {run.answer_mode} / {run.source_count} sources
                  </span>
                </button>
              ))}
            </div>
          </section>

          {error ? <p className="error-banner">{error}</p> : null}
        </aside>

        <section className="panel panel-results">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Workspace</p>
              <h2>{result ? "Research output" : "No run selected"}</h2>
            </div>
            {result ? (
              <div className="action-row">
                <a href={exportHref(result.run_id, "markdown")}>Markdown</a>
                <a href={exportHref(result.run_id, "csv")}>CSV</a>
                <a href={exportHref(result.run_id, "json")}>JSON</a>
              </div>
            ) : null}
          </div>

          {result ? (
            <div className="result-stack">
              <section className="run-metadata">
                <div>
                  <span>Run</span>
                  <strong>{result.run_id}</strong>
                </div>
                <div>
                  <span>Corpus version</span>
                  <strong>{result.corpus_version_id}</strong>
                </div>
                <div>
                  <span>Source documents</span>
                  <strong>{result.corpus_stats.source_document_count}</strong>
                </div>
                <div>
                  <span>Indexed chunks</span>
                  <strong>{result.corpus_stats.chunk_count}</strong>
                </div>
              </section>

              <section className="answer-card">
                <div className="section-heading">
                  <p className="section-label">Answer</p>
                  <div className="feedback-controls">
                    <button type="button" onClick={() => void submitFeedback("up")}>
                      Useful
                    </button>
                    <button type="button" onClick={() => void submitFeedback("down")}>
                      Review
                    </button>
                  </div>
                </div>
                <p className="answer-text">{result.answer}</p>
                {feedbackStatus ? <p className="feedback-status">{feedbackStatus}</p> : null}
              </section>

              {result.claims.length ? (
                <section className="claims-section">
                  <div className="section-heading">
                    <p className="section-label">Claims</p>
                    <span>{result.claims.length} verified</span>
                  </div>
                  <div className="claim-list">
                    {result.claims.map((claim) => (
                      <article key={claim.claim_text} className="claim-card">
                        <div className="claim-topline">
                          <span className="source-type">{claim.support_status}</span>
                          <span className="source-rank">{claim.confidence}</span>
                        </div>
                        <p>{claim.claim_text}</p>
                        <div className="source-meta">
                          {claim.supporting_source_ids.map((sourceId) => (
                            <span key={sourceId} className="meta-chip">
                              {sourceId}
                            </span>
                          ))}
                        </div>
                        {claim.supporting_quotes.map((quote) => (
                          <blockquote key={`${quote.source_id}-${quote.quote}`}>
                            <strong>{quote.source_id}</strong>
                            {quote.quote}
                          </blockquote>
                        ))}
                      </article>
                    ))}
                  </div>
                </section>
              ) : null}

              <section className="evidence-section">
                <div className="section-heading">
                  <p className="section-label">Evidence table</p>
                  <span>{result.sources.length} items</span>
                </div>
                <div className="evidence-table-wrap">
                  <table className="evidence-table">
                    <thead>
                      <tr>
                        <th>Source</th>
                        <th>Type</th>
                        <th>Page</th>
                        <th>Rank</th>
                        <th>Snippet</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.sources.map((source) => (
                        <tr key={source.source_id} onClick={() => void loadSource(source)}>
                          <td>{source.title}</td>
                          <td>{source.source_type}</td>
                          <td>{formatPages(source.metadata)}</td>
                          <td>{formatRank(source.metadata)}</td>
                          <td>{source.snippet}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>

              <section className="source-viewer">
                <div className="section-heading">
                  <p className="section-label">Source viewer</p>
                  <span>{sourceLoading ? "loading" : selectedSource?.source_id ?? "none"}</span>
                </div>
                {selectedSource ? (
                  <article className="source-detail">
                    <div className="source-topline">
                      <span className="source-type">{selectedSource.source_type}</span>
                      {selectedSource.page_start ? (
                        <span className="source-rank">{formatDetailPages(selectedSource)}</span>
                      ) : null}
                    </div>
                    <h3>{selectedSource.title}</h3>
                    <div className="source-meta">
                      {renderMetaChip("chunk", selectedSource.chunk_id)}
                      {renderMetaChip("section", selectedSource.section_title)}
                      {renderMetaChip("document", selectedSource.source_document_id)}
                    </div>
                    <p className="source-text">
                      {renderHighlightedText(selectedSource.text, selectedSource.highlights)}
                    </p>
                  </article>
                ) : (
                  <div className="empty-state compact-empty">
                    <p className="section-label">Source detail</p>
                    <h3>{result.sources.length ? "Select evidence to inspect." : "No evidence returned."}</h3>
                  </div>
                )}
              </section>

              <section className="metrics-grid">
                {result.evaluation.map((metric) => (
                  <article key={metric.metric} className="metric-card">
                    <div className="metric-row">
                      <span>{metric.metric}</span>
                      <strong>{Math.round(metric.score * 100)}%</strong>
                    </div>
                    <div className="metric-bar">
                      <div style={{ width: `${metric.score * 100}%` }} />
                    </div>
                    <p>{metric.rationale}</p>
                  </article>
                ))}
              </section>

              <section className="trace-section">
                <div className="section-heading">
                  <p className="section-label">Execution trace</p>
                </div>
                <div className="trace-list">
                  {result.execution_trace.map((step) => (
                    <div key={step} className="trace-item">
                      <span className="trace-dot" />
                      <code>{step}</code>
                    </div>
                  ))}
                </div>
              </section>
            </div>
          ) : (
            <div className="empty-state">
              <p className="section-label">Workspace</p>
              <h3>Submit or reopen a research run.</h3>
            </div>
          )}
        </section>
      </section>
    </main>
  );
}

function sourceToDetail(runId: string, source: SourceItem): RunSourceDetail {
  return {
    run_id: runId,
    source_id: source.source_id,
    title: source.title,
    source_type: source.source_type,
    snippet: source.snippet,
    text: source.snippet,
    source_document_id: valueToString(source.metadata.source_document_id) ?? null,
    chunk_id: valueToString(source.metadata.chunk_id) ?? source.source_id,
    page_start: valueToNumber(source.metadata.page_start),
    page_end: valueToNumber(source.metadata.page_end),
    section_title: valueToString(source.metadata.section_title) ?? null,
    highlights: [source.snippet],
    metadata: source.metadata,
  };
}

function exportHref(runId: string, format: "markdown" | "csv" | "json") {
  return `${API_BASE_URL}/runs/${runId}/export?format=${format}`;
}

function renderMetaChip(label: string, value: string | number | null | undefined) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  return (
    <span className="meta-chip">
      {label}: {value}
    </span>
  );
}

function formatPages(metadata: Record<string, MetadataValue>) {
  const pageStart = valueToNumber(metadata.page_start);
  const pageEnd = valueToNumber(metadata.page_end);
  if (!pageStart) {
    return "n/a";
  }
  return pageEnd && pageEnd !== pageStart ? `${pageStart}-${pageEnd}` : `${pageStart}`;
}

function formatDetailPages(source: RunSourceDetail) {
  if (!source.page_start) {
    return "page n/a";
  }
  return source.page_end && source.page_end !== source.page_start
    ? `pages ${source.page_start}-${source.page_end}`
    : `page ${source.page_start}`;
}

function formatRank(metadata: Record<string, MetadataValue>) {
  return (
    valueToString(metadata.global_rerank_rank) ??
    valueToString(metadata.hybrid_rank) ??
    valueToString(metadata.retrieval_rank) ??
    "n/a"
  );
}

function valueToString(value: MetadataValue | undefined) {
  if (value === null || value === undefined || typeof value === "boolean") {
    return undefined;
  }
  return String(value);
}

function valueToNumber(value: MetadataValue | undefined) {
  if (typeof value === "number") {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function renderHighlightedText(text: string, highlights: string[]) {
  const highlight = highlights.find((candidate) => candidate && text.includes(candidate));
  if (!highlight) {
    return text;
  }
  const [before, afterFirst] = text.split(highlight);
  const after = afterFirst ?? "";
  return (
    <>
      {before}
      <mark>{highlight}</mark>
      {after}
    </>
  );
}

export default App;
