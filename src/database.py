from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    create_engine,
    DateTime,
    Float,
    Integer,
    String,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    sessionmaker,
)

from config import DATABASE_URL


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    customer_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    prediction: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    risk: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def init_db() -> None:
    Base.metadata.create_all(engine)


def log_prediction(
    customer_id: str | None,
    probability: float,
    prediction: str,
    risk: str,
) -> None:
    with SessionLocal() as session:
        session.add(
            PredictionLog(
                customer_id=customer_id,
                probability=probability,
                prediction=prediction,
                risk=risk,
            )
        )
        session.commit()
