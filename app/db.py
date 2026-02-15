from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    create_engine,
    String,
    Integer,
    Float,
    DateTime,
    ForeignKey,
    JSON,
    select,
    desc,
    inspect,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session

from app.config import Settings


class Base(DeclarativeBase):
    pass


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    provider: Mapped[str] = mapped_column(String(32))
    model_version: Mapped[str] = mapped_column(String(64))
    market_sentiment: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    scores: Mapped[list["Score"]] = relationship(back_populates="scan", cascade="all, delete-orphan")


class Score(Base):
    __tablename__ = "scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scan_id: Mapped[int] = mapped_column(ForeignKey("scans.id", ondelete="CASCADE"), index=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)

    # Primary target: +5% intraday
    prob_5pct: Mapped[float] = mapped_column(Float)
    prob_model: Mapped[float] = mapped_column(Float)
    score: Mapped[int] = mapped_column(Integer)

    # Secondary target: +2% intraday
    prob_2pct: Mapped[float] = mapped_column(Float, default=0.0)
    prob_model_2pct: Mapped[float] = mapped_column(Float, default=0.0)
    score_2pct: Mapped[int] = mapped_column(Integer, default=0)

    features: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    reasons: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    scan: Mapped[Scan] = relationship(back_populates="scores")


def _normalize_database_url(url: str) -> str:
    # Render commonly provides postgres://...; SQLAlchemy expects postgresql+psycopg://...
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def get_engine(settings: Settings):
    if settings.database_url:
        db_url = _normalize_database_url(settings.database_url)
        return create_engine(db_url, pool_pre_ping=True)
    # Local fallback
    return create_engine(f"sqlite:///{settings.db_path}", connect_args={"check_same_thread": False})


def _migrate_scores_table(engine) -> None:
    """Best-effort additive migration for existing installs."""
    try:
        insp = inspect(engine)
        if "scores" not in insp.get_table_names():
            return
        cols = {c["name"] for c in insp.get_columns("scores")}
        stmts = []
        if "prob_2pct" not in cols:
            stmts.append("ALTER TABLE scores ADD COLUMN prob_2pct FLOAT")
        if "prob_model_2pct" not in cols:
            stmts.append("ALTER TABLE scores ADD COLUMN prob_model_2pct FLOAT")
        if "score_2pct" not in cols:
            stmts.append("ALTER TABLE scores ADD COLUMN score_2pct INTEGER")

        if not stmts:
            return
        with engine.begin() as conn:
            for s in stmts:
                conn.execute(text(s))
    except Exception:
        # If anything goes wrong, don't kill the app; new deployments will have fresh tables.
        return


def ensure_db(engine) -> None:
    Base.metadata.create_all(engine)
    _migrate_scores_table(engine)


def insert_scan(engine, provider: str, model_version: str, market_sentiment: dict[str, Any]) -> int:
    with Session(engine) as session:
        scan = Scan(
            run_at_utc=datetime.now(timezone.utc),
            provider=provider,
            model_version=model_version,
            market_sentiment=market_sentiment or {},
        )
        session.add(scan)
        session.commit()
        return int(scan.id)


def insert_scores(engine, scan_id: int, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with Session(engine) as session:
        for r in rows:
            session.add(
                Score(
                    scan_id=scan_id,
                    symbol=str(r.get("symbol")),
                    prob_5pct=float(r.get("prob_5pct") or 0.0),
                    prob_model=float(r.get("prob_model") or r.get("prob_5pct") or 0.0),
                    score=int(r.get("score") or 0),
                    prob_2pct=float(r.get("prob_2pct") or 0.0),
                    prob_model_2pct=float(r.get("prob_model_2pct") or r.get("prob_2pct") or 0.0),
                    score_2pct=int(r.get("score_2pct") or 0),
                    features=r.get("features") or {},
                    reasons=r.get("reasons") or {},
                )
            )
        session.commit()


def get_latest_scan(engine) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    with Session(engine) as session:
        scan = session.execute(select(Scan).order_by(desc(Scan.run_at_utc)).limit(1)).scalar_one_or_none()
        if scan is None:
            return None, []

        scores = (
            session.execute(
                select(Score)
                .where(Score.scan_id == scan.id)
                .order_by(desc(Score.prob_5pct))
                .limit(200)
            )
            .scalars()
            .all()
        )

        scan_dict = {
            "id": int(scan.id),
            "run_at_utc": scan.run_at_utc.isoformat(),
            "provider": scan.provider,
            "model_version": scan.model_version,
            "market_sentiment": scan.market_sentiment or {},
        }

        score_rows: list[dict[str, Any]] = []
        for s in scores:
            score_rows.append(
                {
                    "symbol": s.symbol,
                    "prob_5pct": float(s.prob_5pct),
                    "prob_model": float(s.prob_model),
                    "score": int(s.score),
                    "prob_2pct": float(getattr(s, "prob_2pct", 0.0) or 0.0),
                    "prob_model_2pct": float(getattr(s, "prob_model_2pct", 0.0) or 0.0),
                    "score_2pct": int(getattr(s, "score_2pct", 0) or 0),
                    "features": s.features or {},
                    "reasons": s.reasons or {},
                }
            )
        return scan_dict, score_rows
