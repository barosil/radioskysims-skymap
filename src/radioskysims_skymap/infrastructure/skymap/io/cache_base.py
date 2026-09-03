"""
Base abstratas para cache de SkyMap.

Define contratos para diferentes implementações de cache
(memória, disco, remoto, etc.) de forma desacoplada.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class CacheKey:
    """
    Chave de cache com metadados.

    Permite rastrear:
    - Hash do conteúdo
    - Metadados (freq, nside, etc.)
    - TTL (time-to-live)
    """

    def __init__(self, key_hash: str, metadata: dict[str, Any] | None = None):
        self.key_hash = key_hash
        self.metadata = metadata or {}

    def __str__(self) -> str:
        return self.key_hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CacheKey):
            return NotImplemented
        return self.key_hash == other.key_hash

    def __hash__(self) -> int:
        return hash(self.key_hash)


class CacheEntry(Generic[T]):
    """
    Entrada de cache com valor e metadados.
    """

    def __init__(self, value: T, key: CacheKey, metadata: dict[str, Any] | None = None):
        self.value = value
        self.key = key
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"CacheEntry(key={self.key.key_hash[:8]}..., size={len(str(self.value))})"


class CacheBackend(ABC, Generic[T]):
    """
    Interface abstrata para implementações de cache.

    Permite múltiplas estratégias:
    - InMemoryCache: cache em RAM
    - DiskCache: cache em filesystem
    - HybridCache: combina memória + disco
    """

    @abstractmethod
    def get(self, key: CacheKey) -> T | None:
        """
        Recupera valor do cache.

        Args:
            key: Chave de cache

        Returns:
            Valor em cache ou None se não encontrado
        """
        ...

    @abstractmethod
    def set(self, key: CacheKey, value: T) -> None:
        """
        Armazena valor no cache.

        Args:
            key: Chave de cache
            value: Valor a armazenar
        """
        ...

    @abstractmethod
    def exists(self, key: CacheKey) -> bool:
        """
        Verifica se chave existe no cache.

        Args:
            key: Chave de cache

        Returns:
            True se chave existe, False caso contrário
        """
        ...

    @abstractmethod
    def delete(self, key: CacheKey) -> bool:
        """
        Remove chave do cache.

        Args:
            key: Chave de cache

        Returns:
            True se removido, False se não existia
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """
        Limpa todo o cache.
        """
        ...

    @abstractmethod
    def size(self) -> int:
        """
        Retorna número de itens em cache.

        Returns:
            Número de entradas armazenadas
        """
        ...


class CacheKeyGenerator(ABC):
    """
    Interface para gerar chaves de cache a partir de parâmetros.

    Permite diferentes estratégias de hashing e normalização.
    """

    @abstractmethod
    def generate(self, **params: Any) -> CacheKey:
        """
        Gera chave de cache a partir de parâmetros.

        Args:
            **params: Parâmetros para gerar chave

        Returns:
            CacheKey gerada
        """
        ...
