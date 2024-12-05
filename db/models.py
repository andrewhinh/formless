"""Assumes "DB_URL" and "in_prod" are set env vars in Modal."""

import datetime

from sqlmodel import Field, SQLModel, create_engine
from sqlmodel import Session as DBSession

from utils import DB_URI, IN_PROD

engine = create_engine(
    url=DB_URI,
    echo=not IN_PROD,
)


def get_db_session() -> DBSession:
    return DBSession(engine)


### generations
class GenBase(SQLModel):
    request_at: datetime.datetime = Field(default_factory=datetime.datetime.now(datetime.UTC))
    image_url: str = None
    image_file: str = None
    question: str = None
    failed: bool = False
    response: str = None
    session_id: str = None


class Gen(GenBase, table=True):
    id: int = Field(default=None, primary_key=True)


class GenCreate(GenBase):
    pass


class GenRead(GenBase):
    id: int


class GenUpdate(SQLModel):
    image_url: str = None
    image_file: str = None
    question: str = None
    failed: bool = False
    response: str = None


### api keys
class ApiKeyBase(SQLModel):
    key: str = None
    granted_at: datetime.datetime = Field(default_factory=datetime.datetime.now(datetime.UTC))
    session_id: str = None


class ApiKey(ApiKeyBase, table=True):
    id: int = Field(default=None, primary_key=True)


class ApiKeyCreate(ApiKeyBase):
    pass


class ApiKeyRead(ApiKeyBase):
    id: int


class ApiKeyUpdate(SQLModel):
    key: str = None
    granted_at: datetime.datetime = Field(default_factory=datetime.datetime.now(datetime.UTC))


### global balance
init_balance = 100


class GlobalBalanceBase(SQLModel):
    balance: int = init_balance


class GlobalBalance(GlobalBalanceBase, table=True):
    id: int = Field(default=None, primary_key=True)


class GlobalBalanceCreate(GlobalBalanceBase):
    pass


class GlobalBalanceRead(GlobalBalanceBase):
    id: int


class GlobalBalanceUpdate(SQLModel):
    balance: int = None
