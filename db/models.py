"""Assumes "DB_URL" and "in_prod" are set env vars in Modal."""

import datetime

from sqlmodel import Field, SQLModel


### generations
class GenBase(SQLModel):
    request_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    image_url: str | None = None
    image_file: str | None = None
    question: str | None = None
    failed: bool | None = False
    response: str | None = None
    session_id: str = None


class Gen(GenBase, table=True):
    id: int = Field(default=None, primary_key=True)


class GenCreate(GenBase):
    pass


class GenRead(GenBase):
    id: int = None


### api keys
class ApiKeyBase(SQLModel):
    key: str = None
    granted_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    session_id: str = None


class ApiKey(ApiKeyBase, table=True):
    id: int = Field(default=None, primary_key=True)


class ApiKeyCreate(ApiKeyBase):
    pass


class ApiKeyRead(ApiKeyBase):
    id: int = None


### global balance
init_balance = 100


class GlobalBalanceBase(SQLModel):
    balance: int = init_balance


class GlobalBalance(GlobalBalanceBase, table=True):
    id: int = Field(default=None, primary_key=True)


class GlobalBalanceCreate(GlobalBalanceBase):
    pass


class GlobalBalanceRead(GlobalBalanceBase):
    id: int = None


class GlobalBalanceUpdate(SQLModel):
    balance: int = None
