CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS source_documents (
    source_document_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    publication_year INTEGER NULL,
    doi TEXT NULL,
    source_url TEXT NULL,
    source_type TEXT NOT NULL DEFAULT 'internal',
    checksum TEXT NULL,
    ingestion_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    parser_version TEXT NOT NULL DEFAULT 'prototype',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    quality_flags JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS document_pages (
    source_document_id TEXT NOT NULL REFERENCES source_documents(source_document_id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    extracted_text TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (source_document_id, page_number)
);

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id TEXT PRIMARY KEY,
    source_document_id TEXT NOT NULL REFERENCES source_documents(source_document_id) ON DELETE CASCADE,
    parent_chunk_id TEXT NULL,
    section_title TEXT NULL,
    page_start INTEGER NULL,
    page_end INTEGER NULL,
    text_start_offset INTEGER NULL,
    text_end_offset INTEGER NULL,
    raw_text TEXT NOT NULL,
    contextualized_embedding_text TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR(__EMBEDDING_DIMENSIONS__) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding
    ON document_chunks
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_document_chunks_source_document_id
    ON document_chunks (source_document_id);

CREATE INDEX IF NOT EXISTS idx_document_chunks_fts
    ON document_chunks
    USING gin (to_tsvector('english', contextualized_embedding_text));

CREATE TABLE IF NOT EXISTS corpus_versions (
    corpus_version_id TEXT PRIMARY KEY,
    source_document_count INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    checksum TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS research_runs (
    run_id TEXT PRIMARY KEY,
    corpus_version_id TEXT NOT NULL REFERENCES corpus_versions(corpus_version_id),
    question TEXT NOT NULL,
    classification TEXT NULL,
    selected_tools TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    model TEXT NULL,
    answer_mode TEXT NOT NULL DEFAULT 'detailed',
    answer TEXT NOT NULL DEFAULT '',
    evaluation_scores JSONB NOT NULL DEFAULT '[]'::jsonb,
    execution_trace JSONB NOT NULL DEFAULT '[]'::jsonb,
    latency_ms INTEGER NULL,
    cost_estimate_usd NUMERIC(12, 6) NULL,
    error TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS run_feedback (
    feedback_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
    rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    comment TEXT NULL,
    corrected_answer TEXT NULL,
    add_to_eval BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS run_sources (
    run_id TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
    source_id TEXT NOT NULL,
    source_document_id TEXT NULL,
    chunk_id TEXT NULL,
    source_type TEXT NOT NULL,
    title TEXT NOT NULL,
    snippet TEXT NOT NULL,
    score NUMERIC NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (run_id, source_id)
);

CREATE TABLE IF NOT EXISTS claim_evidence_links (
    claim_evidence_link_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
    claim_text TEXT NOT NULL,
    source_id TEXT NOT NULL,
    quote TEXT NOT NULL,
    support_label TEXT NOT NULL DEFAULT 'supporting',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS research_documents (
    document_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'internal',
    source_url TEXT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR(__EMBEDDING_DIMENSIONS__) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_research_documents_embedding
    ON research_documents
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_research_documents_fts
    ON research_documents
    USING gin (to_tsvector('english', title || ' ' || content));
