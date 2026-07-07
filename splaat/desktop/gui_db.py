import sqlalchemy
from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.orm import declarative_base

from splaat.db import PathType

SplaatSqlBase = declarative_base()


class Corpus(SplaatSqlBase):
    __tablename__ = "corpus"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False, index=True)
    path = Column(PathType, nullable=False, index=True, unique=True)
    current = Column(Boolean, nullable=False, default=False, index=True)
    last_used = Column(DateTime, nullable=False, server_default=sqlalchemy.func.now(), index=True)
