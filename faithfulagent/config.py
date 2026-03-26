"""Per-check configuration models and defaults."""

from pydantic import BaseModel, ConfigDict


class NumericCheckConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tolerance: float = 0.0
    allow_rounding: bool = True
    max_rounding_deviation: float = 10.0


class StringCheckConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_sensitive: bool = False
    fuzzy_threshold: float = 0.85


class UnitCheckConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allow_conversion: bool = True


class DerivedCheckConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tolerance: float = 0.01


class CrossClaimCheckConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    sum_tolerance: float = 0.01


class VerifyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    numeric: NumericCheckConfig = NumericCheckConfig()
    string: StringCheckConfig = StringCheckConfig()
    unit: UnitCheckConfig = UnitCheckConfig()
    derived: DerivedCheckConfig = DerivedCheckConfig()
    cross_claim: CrossClaimCheckConfig = CrossClaimCheckConfig()

    @classmethod
    def from_dict(cls, raw: dict) -> "VerifyConfig":
        return cls(
            numeric=NumericCheckConfig(**raw.get("numeric", {})),
            string=StringCheckConfig(**raw.get("string", {})),
            unit=UnitCheckConfig(**raw.get("unit", {})),
            derived=DerivedCheckConfig(**raw.get("derived", {})),
            cross_claim=CrossClaimCheckConfig(**raw.get("cross_claim", {})),
        )
