from pydantic import BaseModel


class ClaimRequest(BaseModel):
    description: str


class PredictionResponse(BaseModel):
    insurance_type: str
    insurance_confidence: float

    claim_type: str
    claim_confidence: float

    severity: str
    severity_confidence: float

    department: str
    department_confidence: float

    recommendation: str
