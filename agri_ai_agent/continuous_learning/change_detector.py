from datetime import datetime

from agri_ai_agent.continuous_learning.repository_registry import RepositoryRegistry
from agri_ai_agent.continuous_learning.version_history import VersionHistory

CHANGE_SOURCE_DATA = "data"
CHANGE_SOURCE_NONE = "none"


class ChangeSet:
    def __init__(self):
        self.changed_repos: list[str] = []
        self.new_datasets: list[str] = []
        self.changed_observations: list[str] = []
        self.changed_features: list[str] = []
        self.new_model_versions: list[str] = []
        self.changed_reckoner: bool = False
        self.change_source: str = CHANGE_SOURCE_NONE
        self.timestamp: str = datetime.now().isoformat()

    @property
    def has_changes(self) -> bool:
        return bool(
            self.changed_repos
            or self.new_datasets
            or self.changed_observations
            or self.changed_features
            or self.new_model_versions
            or self.changed_reckoner
        )

    @property
    def has_data_changes(self) -> bool:
        return bool(self.new_datasets or self.changed_observations or self.changed_features)

    @property
    def has_model_changes(self) -> bool:
        return bool(self.new_model_versions)

    def merge(self, other: "ChangeSet"):
        for repo in other.changed_repos:
            if repo not in self.changed_repos:
                self.changed_repos.append(repo)
        for ds in other.new_datasets:
            if ds not in self.new_datasets:
                self.new_datasets.append(ds)
        for obs in other.changed_observations:
            if obs not in self.changed_observations:
                self.changed_observations.append(obs)
        for feat in other.changed_features:
            if feat not in self.changed_features:
                self.changed_features.append(feat)
        for mv in other.new_model_versions:
            if mv not in self.new_model_versions:
                self.new_model_versions.append(mv)
        if other.changed_reckoner:
            self.changed_reckoner = True
        if other.has_changes:
            self.change_source = other.change_source

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "has_changes": self.has_changes,
            "has_data_changes": self.has_data_changes,
            "has_model_changes": self.has_model_changes,
            "change_source": self.change_source,
            "changed_repos": self.changed_repos,
            "new_datasets": self.new_datasets,
            "changed_observations": self.changed_observations,
            "changed_features": self.changed_features,
            "new_model_versions": self.new_model_versions,
            "changed_reckoner": self.changed_reckoner,
        }


class ChangeDetector:
    def __init__(self, registry: RepositoryRegistry, history: VersionHistory):
        self._registry = registry
        self._history = history

    def check_all(self) -> ChangeSet:
        changes = ChangeSet()
        repos = self._registry.list_enabled()
        if not repos:
            return changes

        for repo in repos:
            repo_changes = self._check_repo(repo)
            changes.merge(repo_changes)

        if changes.has_changes:
            changes.change_source = CHANGE_SOURCE_DATA
        return changes

    def check_repo(self, name: str) -> ChangeSet:
        repo = self._registry.get(name)
        if not repo:
            return ChangeSet()
        return self._check_repo(repo)

    def _check_repo(self, repo: dict) -> ChangeSet:
        changes = ChangeSet()
        name = repo.get("name", "")
        if not repo.get("enabled", 0):
            return changes

        last_version = repo.get("last_version", "") or ""
        available = self._probe_version(name)
        if available and available != last_version:
            changes.changed_repos.append(name)
            changes.change_source = CHANGE_SOURCE_DATA
        return changes

    def _probe_version(self, repo_name: str) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")
