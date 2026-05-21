import { FormEvent, startTransition, useDeferredValue, useState } from "react";

type SourceItem = {
  source_id: string;
  title: string;
  snippet: string;
  source_type: string;
  url: string | null;
  metadata: Record<string, string | number | null>;
};

type EvaluationScore = {
  metric: string;
  score: number;
  rationale: string;
};

type CorpusStats = {
  source_document_count: number;
  chunk_count: number;
  corpus_version_id: string;
};

type ResearchResponse = {
  run_id: string;
  corpus_version_id: string;
  corpus_stats: CorpusStats;
  question: string;
  answer: string;
  sources: SourceItem[];
  evaluation: EvaluationScore[];
  execution_trace: string[];
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

const QUICK_PROMPTS = [
  "What are the main challenges of RAG according to the internal corpus?",
  "What is the difference between LangChain and LangGraph in the indexed documents?",
  "How many indexed chunks are stored in the project corpus?",
];

function App() {
  const [question, setQuestion] = useState(QUICK_PROMPTS[0]);
  const [topK, setTopK] = useState(4);
  const [result, setResult] = useState<ResearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const deferredQuestion = useDeferredValue(question);

  const loweredQuestion = deferredQuestion.toLowerCase();
  let questionLens = "General research";
  if (/(how many|quantos|count|total|sum)/.test(loweredQuestion)) {
    questionLens = "Quantitative query";
  } else if (/(compar|differenc|diferenc|vs)/.test(loweredQuestion)) {
    questionLens = "Comparative question";
  } else if (/(challenge|desafio|risk|limita)/.test(loweredQuestion)) {
    questionLens = "Risk analysis";
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!question.trim()) {
      setError("Enter a question to start the research workflow.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/research`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question,
          top_k: topK,
        }),
      });

      if (!response.ok) {
        throw new Error(`Research request failed (${response.status})`);
      }

      const payload = (await response.json()) as ResearchResponse;
      startTransition(() => {
        setResult(payload);
      });
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

  return (
    <main className="shell">
      <section className="workspace">
        <div className="panel panel-form">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Composer</p>
              <h2>New research run</h2>
            </div>
            <div className="lens-pill">{questionLens}</div>
          </div>

          <form className="research-form" onSubmit={handleSubmit}>
            <label className="field">
              <span>Question</span>
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Example: What are the main challenges of RAG according to the internal corpus?"
                rows={7}
              />
            </label>

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

          {error ? <p className="error-banner">{error}</p> : null}
        </div>

        <div className="panel panel-results">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Output</p>
              <h2>Execution result</h2>
            </div>
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
                <p className="section-label">Answer</p>
                <p className="answer-text">{result.answer}</p>
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

              <section className="sources-section">
                <div className="section-heading">
                  <p className="section-label">Sources</p>
                  <span>{result.sources.length} items</span>
                </div>
                <div className="source-list">
                  {result.sources.map((source) => (
                    <article key={source.source_id} className="source-card">
                      <div className="source-topline">
                        <span className="source-type">{source.source_type}</span>
                        {source.metadata?.retrieval_rank ? (
                          <span className="source-rank">rank {source.metadata.retrieval_rank}</span>
                        ) : null}
                      </div>
                      <h3>{source.title}</h3>
                      <p>{source.snippet}</p>
                      <div className="source-meta">
                        {renderMetaChip("file", source.metadata?.source_file)}
                        {renderMetaChip("section", source.metadata?.section_title)}
                        {renderPageChip(source.metadata)}
                        {renderMetaChip("domain", source.metadata?.domain)}
                      </div>
                      {source.url ? (
                        <a href={source.url} target="_blank" rel="noreferrer">
                          Open source
                        </a>
                      ) : null}
                    </article>
                  ))}
                </div>
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
              <p className="section-label">Ready to research</p>
              <h3>Submit a question from the panel on the left.</h3>
              <p>
                The final answer will appear here with sources, heuristic scoring, and
                execution trace.
              </p>
            </div>
          )}
        </div>
      </section>
    </main>
  );
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

function renderPageChip(metadata: Record<string, string | number | null>) {
  const pageStart = metadata?.page_start;
  const pageEnd = metadata?.page_end;

  if (typeof pageStart !== "number") {
    return null;
  }

  return (
    <span className="meta-chip">
      {pageEnd && pageEnd !== pageStart ? `pages: ${pageStart}-${pageEnd}` : `page: ${pageStart}`}
    </span>
  );
}

export default App;
