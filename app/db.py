"""Task 5 - database trail (SQLAlchemy: SQLite locally, Postgres in Compose/cloud)."""
import datetime as dt

from sqlalchemy import DateTime, Float, Integer, String, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None))
    filename: Mapped[str] = mapped_column(String(255))
    file_sha256: Mapped[str] = mapped_column(String(64))
    label: Mapped[str] = mapped_column(String(20))  # what was decided
    prob_ai: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)  # how confidently
    threshold: Mapped[float] = mapped_column(Float, nullable=True)
    model_name: Mapped[str] = mapped_column(String(100))  # by which model
    model_version: Mapped[str] = mapped_column(String(50))
    latency_ms: Mapped[float] = mapped_column(Float, nullable=True)


class Database:
    def __init__(self, url):
        self.engine = create_engine(url, pool_pre_ping=True)
        Base.metadata.create_all(self.engine)

    def add(self, **kw):
        with Session(self.engine) as s:
            s.add(Prediction(**kw))
            s.commit()

    def recent(self, limit=100):
        with Session(self.engine) as s:
            rows = s.scalars(select(Prediction).order_by(Prediction.id.desc()).limit(limit)).all()
            return [r for r in rows]

    def ping(self):
        with self.engine.connect() as c:
            c.execute(text("SELECT 1"))
        return True
