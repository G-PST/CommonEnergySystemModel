"""
Core data handling utilities for working with Database objects.
"""
from typing import Type, Optional, Dict, Any, List
from generated.cesm import Database, Entity


class DataHandler:
    """
    Utility class for working with Database objects.
    Provides generic operations that work across all entity types.
    """

    def __init__(self, database: Database):
        """
        Initialize the data handler with a Database object.

        Args:
            database: The Database object to work with
        """
        self.database = database

    def get_entity_collections(self) -> Dict[str, List[Any]]:
        """
        Get all entity collections from the database.

        Returns:
            Dict mapping collection names to entity lists
        """
        collections = {}

        # Get all attributes that contain entity collections
        for attr_name in dir(self.database):
            if not attr_name.startswith('_') and attr_name != 'id':
                attr_value = getattr(self.database, attr_name)
                if isinstance(attr_value, list):
                    collections[attr_name] = attr_value

        return collections

    def count_entities(self) -> Dict[str, int]:
        """
        Count entities in each collection.

        Returns:
            Dict mapping collection names to entity counts
        """
        collections = self.get_entity_collections()
        return {name: len(entities) for name, entities in collections.items()}

    def find_entity_by_id(self, entity_id: int) -> Optional[Entity]:
        """
        Find any entity by its ID across all collections.

        Args:
            entity_id: The ID to search for

        Returns:
            The entity if found, None otherwise
        """
        collections = self.get_entity_collections()

        for entities in collections.values():
            for entity in entities:
                if hasattr(entity, 'id') and entity.id == entity_id:
                    return entity

        return None

    def find_entity_by_name(self, name: str) -> Optional[Entity]:
        """
        Find any entity by its name across all collections.

        Args:
            name: The name to search for

        Returns:
            The entity if found, None otherwise
        """
        collections = self.get_entity_collections()

        for entities in collections.values():
            for entity in entities:
                if hasattr(entity, 'name') and entity.name == name:
                    return entity

        return None

    def validate_database(self) -> List[str]:
        """
        Perform basic validation checks on the database.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check for duplicate IDs across all entities
        seen_ids = set()
        collections = self.get_entity_collections()

        for collection_name, entities in collections.items():
            for entity in entities:
                if hasattr(entity, 'id'):
                    if entity.id in seen_ids:
                        errors.append(f"Duplicate ID {entity.id} found in {collection_name}")
                    seen_ids.add(entity.id)

        # Check for duplicate names within each collection
        for collection_name, entities in collections.items():
            names = [entity.name for entity in entities if hasattr(entity, 'name')]
            duplicates = [name for name in names if names.count(name) > 1]
            if duplicates:
                unique_duplicates = list(set(duplicates))
                errors.append(f"Duplicate names in {collection_name}: {unique_duplicates}")

        return errors
