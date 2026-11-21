"""Pydantic request and response schemas for the 13-feature classifier."""

from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

Outcome = Literal["CKD_RISK", "NORMAL"]


class PredictionRequest(BaseModel):
    """Clinical inputs in the order used by the training dataset."""

    model_config = ConfigDict(extra="forbid")

    bp: float = Field(ge=20, le=200)
    sg: Literal[1.005, 1.010, 1.015, 1.020, 1.025]
    al: int = Field(ge=0, le=5)
    su: int = Field(ge=0, le=5)
    rbc: Literal["normal", "abnormal"]
    bu: float = Field(ge=0, le=500)
    sc: float = Field(ge=0, le=50)
    sod: float = Field(ge=100, le=180)
    pot: float = Field(ge=1, le=10)
    hemo: float = Field(ge=0, le=25)
    wc: float = Field(ge=0, le=100000, validation_alias=AliasChoices("wc", "wbcc"))
    rc: float = Field(ge=0, le=10, validation_alias=AliasChoices("rc", "rbcc"))
    htn: Literal["yes", "no"]


class PredictionResponse(BaseModel):
    outcome: Outcome
