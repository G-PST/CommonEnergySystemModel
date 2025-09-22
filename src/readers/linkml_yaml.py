"""
LinkML YAML reader implementation.
"""
from pathlib import Path
from linkml_runtime.loaders.yaml_loader import YAMLLoader
from core.interfaces import DataReader
from generated.cesm import Database


class LinkMLYAMLReader(DataReader):
    """Reader for LinkML YAML format files."""

    def __init__(self):
        self.loader = YAMLLoader()

    def read(self, source: str | Path) -> Database:
        """
        Read LinkML YAML data and return a Database object.

        Args:
            source: Path to the YAML file

        Returns:
            Database: Populated Database object with entity collections

        Raises:
            FileNotFoundError: If the source file doesn't exist
            ValueError: If the YAML is invalid or doesn't match the schema
        """
        source_path = Path(source)

        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {source_path}")

        try:
            # Load the YAML file and instantiate Database object
            database = self.loader.load(str(source_path), target_class=Database)
            return database

        except Exception as e:
            raise ValueError(f"Failed to load YAML data: {e}") from e