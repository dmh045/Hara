from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

from .common import ReviewStatus, SourceRef


class GuidewordDisposition(str, Enum):
    DOWNSTREAM_CANDIDATE = "DOWNSTREAM_CANDIDATE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NO_CREDIBLE_HAZARD = "NO_CREDIBLE_HAZARD"


@dataclass
class GuidewordAssessment:
    function_id: str
    guideword: str
    applicable: bool
    rationale: str
    sources: list[SourceRef] = field(default_factory=list)
    status: ReviewStatus = ReviewStatus.PENDING
    confidence: float = 0.0
    disposition: GuidewordDisposition | None = None
    # The governed method identifier. Legacy callers fall back to the name.
    guideword_id: str = ""

    def __post_init__(self):
        self.guideword_id = str(self.guideword_id or self.guideword).strip()
        if self.disposition is None:
            self.disposition = (
                GuidewordDisposition.DOWNSTREAM_CANDIDATE
                if self.applicable else GuidewordDisposition.NOT_APPLICABLE
            )
        elif not isinstance(self.disposition, GuidewordDisposition):
            self.disposition = GuidewordDisposition(str(self.disposition))
        if not str(self.function_id).strip():
            raise ValueError("GuidewordAssessment缺少function_id")
        if not str(self.guideword).strip():
            raise ValueError("GuidewordAssessment缺少guideword")
        if not isinstance(self.applicable, bool):
            raise ValueError("GuidewordAssessment.applicable必须为boolean")
        if self.status == ReviewStatus.FINALIZED and not str(self.rationale).strip():
            raise ValueError("FINALIZED GuidewordAssessment缺少rationale")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Guideword confidence必须在0到1之间")
        if (
            self.disposition is GuidewordDisposition.NOT_APPLICABLE
            and self.applicable
        ):
            raise ValueError("NOT_APPLICABLE GuidewordAssessment不得标记applicable=true")
        if (
            self.disposition in {
                GuidewordDisposition.DOWNSTREAM_CANDIDATE,
                GuidewordDisposition.NO_CREDIBLE_HAZARD,
            }
            and not self.applicable
        ):
            raise ValueError(
                f"{self.disposition.value} GuidewordAssessment必须标记applicable=true"
            )

    @property
    def is_semantically_complete(self) -> bool:
        """Whether the assessment can safely drive draft derivation."""
        return bool(str(self.rationale).strip()) and self.disposition is not None

    @property
    def enters_downstream(self) -> bool:
        return self.disposition is GuidewordDisposition.DOWNSTREAM_CANDIDATE


@dataclass
class MalfunctionCandidate:
    malfunction_id: str
    function_id: str
    guideword: str
    description: str
    functional_effect: str
    vehicle_level_hazard: str
    causal_chain: list[str]
    sources: list[SourceRef] = field(default_factory=list)
    status: ReviewStatus = ReviewStatus.PENDING
    confidence: float = 0.0
    model_local_id: str = ""
    component_category: str = ""
    failure_type: str = ""
    # Deterministic, non-semantic taxonomy audit populated after generation.
    selector_resolution: dict[str, object] = field(default_factory=dict)
    # Copied from the accepted GuidewordAssessment; never model-authored.
    guideword_id: str = ""
    # Source-linked analysis premise; not an assertion that the Item specifies
    # this fault or an engineering approval. Empty in historical checkpoints.
    derivation: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Keep legacy task identities stable when no derivation was recorded."""
        value = asdict(self)
        if not self.derivation:
            value.pop("derivation")
        return value

    @property
    def is_analysis_premise(self) -> bool:
        return (
            self.status is ReviewStatus.PENDING
            and self.derivation.get("validation_status") == "SOURCE_LINKED_ANALYSIS_PREMISE"
            and bool(self.sources)
        )

    def __post_init__(self):
        self.guideword_id = str(self.guideword_id or self.guideword).strip()
        required = (
            self.malfunction_id, self.function_id, self.guideword, self.description,
            self.functional_effect, self.vehicle_level_hazard,
        )
        if not all(str(value).strip() for value in required):
            raise ValueError("MalfunctionCandidate缺少必填语义字段")
        if len(self.causal_chain) < 2:
            raise ValueError("MalfunctionCandidate至少需要两段因果链")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Malfunction confidence必须在0到1之间")
