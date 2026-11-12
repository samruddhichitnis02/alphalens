from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class FinancialEventItem(BaseModel):
    event_type: Literal[
        "earnings",
        "guidance_change",
        "mergers_and_acquisitions",
        "executive_change",
        "material_agreement",
        "debt_issuance",
        "other"
    ] = Field(description="Category of the material corporate event.")
    date: Optional[str] = Field(None, description="Date of the event if mentioned in the text (YYYY-MM-DD or descriptive).")
    summary: str = Field(..., description="Concise summary of the corporate event and key financial figures.")
    exact_quote: str = Field(..., description="Exact verbatim supporting sentence directly extracted from the chunk text.")

class FinancialEventList(BaseModel):
    events: List[FinancialEventItem] = Field(default_factory=list)

class BusinessRiskItem(BaseModel):
    category: Literal[
        "regulatory",
        "supply_chain",
        "competition",
        "macro",
        "cybersecurity",
        "litigation",
        "concentration",
        "operational",
        "other"
    ] = Field(description="Category of the business or market risk.")
    description: str = Field(..., description="Concise 1-2 sentence description of the business risk and impact.")
    exact_quote: str = Field(..., description="Exact verbatim supporting sentence directly extracted from the chunk text.")

class BusinessRiskList(BaseModel):
    risks: List[BusinessRiskItem] = Field(default_factory=list)

