"""Shared persistence layer. Agents never talk to each other directly —
they hand off through these tables, which is what lets each stage run in
its own container and be retried independently."""
from datetime import datetime, timezone

from sqlalchemy import (JSON, DateTime, Float, ForeignKey, Integer, String, Text,
                        create_engine)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from copilot.config import get_settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Account(Base):
    """Gold-layer feature row per account (one row, overwritten each run)."""
    __tablename__ = "accounts"
    account_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    segment: Mapped[str] = mapped_column(String(32))          # RFM segment
    supplier_origin: Mapped[str] = mapped_column(String(16))
    features: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RiskScore(Base):
    """Analyst agent output — one row per account per pipeline run."""
    __tablename__ = "risk_scores"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(40), index=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.account_id"), index=True)
    clv_score: Mapped[float] = mapped_column(Float)
    churn_probability: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(8))
    top_factors: Mapped[list] = mapped_column(JSON)
    monitor_signals: Mapped[list] = mapped_column(JSON, default=list)
    # Pilot cohort: "copilot" (gets a draft) or "control" (handled as today, no draft)
    cohort: Mapped[str | None] = mapped_column(String(10), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Outcome(Base):
    """What actually happened to an account, imported from the ERP after the fact
    (e.g. "no order in the 6 months after period_start"). Feeds the Trust metrics."""
    __tablename__ = "outcomes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[str] = mapped_column(String(32), index=True)
    period_start: Mapped[str] = mapped_column(String(10))       # YYYY-MM-DD
    period_end: Mapped[str] = mapped_column(String(10))
    churned: Mapped[int] = mapped_column(Integer)                 # 1 = churned, 0 = retained
    revenue: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(40), default="erp")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Draft(Base):
    """Writer agent output, waiting for human review."""
    __tablename__ = "drafts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(40), index=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.account_id"), index=True)
    action: Mapped[str] = mapped_column(String(64))
    action_rationale: Mapped[str] = mapped_column(Text)
    subject: Mapped[str] = mapped_column(String(300))
    body: Mapped[str] = mapped_column(Text)
    context_used: Mapped[list] = mapped_column(JSON, default=list)
    # pending_review | insufficient_data | approved | rejected | needs_edit
    status: Mapped[str] = mapped_column(String(20), index=True, default="pending_review")
    # What the writer learned from: {"approved_examples": [draft ids], "lessons": [event ids]}
    feedback_used: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReviewEvent(Base):
    """Immutable audit log of every human decision — feeds the writer
    feedback loop and the evaluation metrics."""
    __tablename__ = "review_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    draft_id: Mapped[int] = mapped_column(ForeignKey("drafts.id"), index=True)
    reviewer: Mapped[str] = mapped_column(String(100))
    decision: Mapped[str] = mapped_column(String(20))
    # Required for rejected / needs_edit: a code from copilot.feedback.REASONS
    reason: Mapped[str | None] = mapped_column(String(40), nullable=True)
    # The draft as the writer produced it, kept so (original -> final) pairs can train the writer
    original_body: Mapped[str | None] = mapped_column(Text, nullable=True)
    edited_body: Mapped[str | None] = mapped_column(Text, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)        # the explanation
    seconds_to_decision: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


_engine = create_engine(get_settings().database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(_engine)
    _add_missing_columns()


def _add_missing_columns() -> None:
    """Tiny migration: create_all() never alters existing tables, so databases created by an
    older version (e.g. demo/_data/demo.db) get new nullable columns added here."""
    from sqlalchemy import inspect, text
    insp = inspect(_engine)
    with _engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            existing = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name not in existing and col.nullable:
                    ddl = col.type.compile(dialect=_engine.dialect)
                    conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {col.name} {ddl}'))
