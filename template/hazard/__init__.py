"""Shared runtime components for the annotation-only hazard subnet."""

from .annotation_eval import (
    AnnotationFidelityScorer,
    ConsensusComponents,
    ConsensusScorer,
    FidelityComponents,
    PerMinerAnnotationScore,
    evaluate_round_annotations,
    iou_xyxy,
)
from .dataset_assembler import AdoptionLedger, DatasetAssembler, WinningAnnotation
from .dual_reward import DualFlywheelBreakdown, DualFlywheelRewardComposer
from .golden_injection import InjectionPlan
from .image_corpus import (
    BenchmarkImage,
    GoldenAnnotation,
    GoldenImage,
    ImageCorpus,
    ImageCorpusConfig,
    UnlabeledImage,
)
from .r2_storage import load_r2_credentials_from_env
