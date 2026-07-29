


STAGE_DATASET_REGISTRATION = "dataset_registration"
STAGE_KNOWLEDGE_GRAPH_UPDATE = "knowledge_graph_update"
STAGE_AI_EXTRACTION = "ai_extraction"
STAGE_OBSERVATION_GENERATION = "observation_generation"
STAGE_VALIDATION = "validation"
STAGE_FEATURE_ENGINEERING = "feature_engineering"
STAGE_MODEL_DRIFT_DETECTION = "model_drift_detection"
STAGE_INCREMENTAL_RETRAINING = "incremental_model_retraining"
STAGE_READY_RECKONER_UPDATE = "ready_reckoner_update"

ALL_USER_STAGES = [
    STAGE_DATASET_REGISTRATION,
    STAGE_KNOWLEDGE_GRAPH_UPDATE,
    STAGE_AI_EXTRACTION,
    STAGE_OBSERVATION_GENERATION,
    STAGE_VALIDATION,
    STAGE_FEATURE_ENGINEERING,
    STAGE_MODEL_DRIFT_DETECTION,
    STAGE_INCREMENTAL_RETRAINING,
    STAGE_READY_RECKONER_UPDATE,
]

PIPELINE_STAGE_MAP = {
    STAGE_DATASET_REGISTRATION: [
        "external_data", "dataset_normalization", "dataset_ingestion_bridge",
    ],
    STAGE_KNOWLEDGE_GRAPH_UPDATE: [
        "external_data", "dataset_normalization",
    ],
    STAGE_AI_EXTRACTION: [
        "extraction", "evidence_fusion", "ontology",
        "table_intelligence", "schema_population",
        "knowledge_integration", "knowledge",
    ],
    STAGE_OBSERVATION_GENERATION: [
        "observation_generation",
    ],
    STAGE_VALIDATION: [
        "validation",
    ],
    STAGE_FEATURE_ENGINEERING: [
        "feature_store", "feature",
    ],
    STAGE_MODEL_DRIFT_DETECTION: [],
    STAGE_INCREMENTAL_RETRAINING: [
        "model_selection", "training",
    ],
    STAGE_READY_RECKONER_UPDATE: [
        "prediction", "recommendation", "fuzzy",
        "benchmark", "explainability", "ready_reckoner",
    ],
}


class DependencyGraph:
    def get_affected_stages(self, changes: ChangeSet) -> list[str]:
        affected: list[str] = []

        if not changes.has_changes:
            return affected

        if changes.has_data_changes:
            affected.extend([
                STAGE_DATASET_REGISTRATION,
                STAGE_KNOWLEDGE_GRAPH_UPDATE,
                STAGE_AI_EXTRACTION,
                STAGE_OBSERVATION_GENERATION,
                STAGE_VALIDATION,
                STAGE_FEATURE_ENGINEERING,
            ])

        if changes.has_data_changes or changes.has_model_changes:
            affected.append(STAGE_MODEL_DRIFT_DETECTION)

        if STAGE_MODEL_DRIFT_DETECTION in affected:
            affected.append(STAGE_INCREMENTAL_RETRAINING)

        if STAGE_INCREMENTAL_RETRAINING in affected:
            affected.append(STAGE_READY_RECKONER_UPDATE)

        seen = []
        for s in affected:
            if s not in seen:
                seen.append(s)
        return seen

    def get_pipeline_steps_for_stages(self, user_stages: list[str]) -> list[str]:
        steps = []
        seen = set()
        for stage in user_stages:
            for step in PIPELINE_STAGE_MAP.get(stage, []):
                if step not in seen:
                    steps.append(step)
                    seen.add(step)
        return steps
