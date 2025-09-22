"""
Core interfaces for the energy interoperability framework.
"""
from abc import ABC, abstractmethod
from typing import Any
from generated.cesm import Database


class DataReader(ABC):
    """Abstract base class for reading data into the schema format."""

    @abstractmethod
    def read(self, source: Any) -> Database:
        """
        Read data from a source and return a populated Database object.

        Args:
            source: The data source (file path, URL, etc.)

        Returns:
            Database: A populated Database object with entity collections

        Raises:
            ValueError: If the data cannot be read or is invalid
            FileNotFoundError: If the source file doesn't exist
        """
        pass


class DataWriter(ABC):
    """Abstract base class for writing data from the schema format."""

    @abstractmethod
    def write(self, data: Database, target: Any) -> None:
        """
        Write data from a Database object to a target format.

        Args:
            data: The Database object containing entity collections
            target: The target destination (file path, connection, etc.)

        Raises:
            ValueError: If the data cannot be written
            IOError: If the target cannot be accessed
        """
        pass