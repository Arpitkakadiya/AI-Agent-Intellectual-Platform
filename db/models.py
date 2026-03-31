from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional

from pydantic import BaseModel, Field


class Jurisdiction(BaseModel):
    id: Optional[int] = None
    type: str
    name: str
    parent_id: Optional[int] = None
    state_code: Optional[str] = None
    fips_code: Optional[str] = None


class Regulation(BaseModel):
    id: Optional[int] = None
    jurisdiction_id: int
    domain: str
    category: str
    source_name: str
    url: str
    content: str
    content_hash: str
    version: int
    is_current: bool
    effective_date: Optional[date] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RegulationEmbedding(BaseModel):
    id: Optional[int] = None
    regulation_id: int
    embedding: list[float]
    chunk_text: str


class EmailSubscription(BaseModel):
    id: Optional[int] = None
    email: str
    jurisdiction_id: int
    subscribed_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True


class RegulationUpdate(BaseModel):
    id: Optional[int] = None
    regulation_id: int
    update_summary: str
    affected_jurisdictions: list[int] = Field(default_factory=list)
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class PetPolicy(BaseModel):
    id: Optional[int] = None
    jurisdiction_id: int
    esa_deposit_allowed: bool
    service_animal_fee: bool
    breed_restrictions: list[str] = Field(default_factory=list)
    max_pet_deposit_amount: Optional[Decimal] = None
    source_regulation_id: int


class InsuranceRequirement(BaseModel):
    id: Optional[int] = None
    jurisdiction_id: int
    landlord_can_require: bool
    min_liability_coverage: Optional[Decimal] = None
    tenant_must_show_proof: bool
    notes: Optional[str] = None
    source_regulation_id: int


class RegulationSource(BaseModel):
    """A scrape-target URL with metadata.  Separates 'what to scrape' from 'what was scraped'."""

    id: Optional[int] = None
    jurisdiction_id: int
    source_name: str
    url: str
    domain: str = "housing"
    category: str = "General"
    state_code: Optional[str] = None
    is_active: bool = True
    last_scraped_at: Optional[datetime] = None
    last_error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AppSetting(BaseModel):
    """Persistent key-value pair for feature flags and app configuration."""

    key: str
    value: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# Convenience aliases for JSONB-ish columns if needed elsewhere
JsonDict = dict[str, Any]
