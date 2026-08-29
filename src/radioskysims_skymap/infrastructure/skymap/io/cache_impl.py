"""
Implementações de cache para SkyMap.

Inclui:
- InMemoryCache: Cache em RAM
- DiskCache: Cache em filesystem (preparado para persistência)
- HybridCache: Combina memória + disco com estratégias de fallback
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Final

from .cache_base import CacheBackend, CacheEntry, CacheKey, CacheKeyGenerator

logger = logging.getLogger(__name__)


# ============================================================================
# Key Generator
# ============================================================================

class SkyMapCacheKeyGenerator(CacheKeyGenerator):
    """
    Gera chaves determinísticas para SkyMap baseado em parâmetros.
    
    Normaliza parâmetros e cria hash SHA256 determinístico.
    """

    def generate(self, **params: Any) -> CacheKey:
        """
        Gera chave a partir de parâmetros normalizados.
        
        Exemplo:
            key = generator.generate(freq_ghz=30.0, nside=1, components=["synchrotron"])
        
        Args:
            **params: Parâmetros para SkyMap (freq_ghz, nside, components, etc.)
            
        Returns:
            CacheKey com hash determinístico
        """
        # Normaliza lista/set em estrutura estável
        normalized: dict[str, Any] = {}
        for k, v in sorted(params.items()):
            if isinstance(v, (list, tuple)):
                normalized[k] = sorted(v)
            else:
                normalized[k] = v

        # Serializa deterministicamente
        serialized = json.dumps(normalized, sort_keys=True, default=str).encode()
        key_hash = hashlib.sha256(serialized).hexdigest()

        return CacheKey(key_hash, metadata=params)


# ============================================================================
# In-Memory Cache
# ============================================================================

class InMemoryCache(CacheBackend[bytes]):
    """
    Cache em memória RAM com limite de tamanho.
    
    Implementa LRU (Least Recently Used) para evitar crescimento ilimitado.
    
    Características:
    - Acesso O(1) em média
    - TTL não implementado (cache enquanto em memória)
    - Perfeito para produtos intermediários de vida curta
    """

    def __init__(self, max_size: int = 100):
        """
        Inicializa cache em memória.
        
        Args:
            max_size: Número máximo de entradas antes de descartar (LRU)
        """
        self._max_size = max_size
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._metadata: dict[str, dict[str, Any]] = {}

    def get(self, key: CacheKey) -> bytes | None:
        """Recupera valor e atualiza ordem LRU."""
        key_str = str(key)
        if key_str not in self._cache:
            return None

        # Move para final (mais recente)
        self._cache.move_to_end(key_str)
        return self._cache[key_str]

    def set(self, key: CacheKey, value: bytes) -> None:
        """Armazena valor com política LRU."""
        key_str = str(key)

        # Se já existe, remove para re-adicionar no final
        if key_str in self._cache:
            del self._cache[key_str]
        elif len(self._cache) >= self._max_size:
            # Remove entrada mais antiga (LRU)
            removed_key, _ = self._cache.popitem(last=False)
            self._metadata.pop(removed_key, None)
            logger.debug(f"Cache LRU eviction: {removed_key[:8]}...")

        self._cache[key_str] = value
        self._metadata[key_str] = key.metadata

    def exists(self, key: CacheKey) -> bool:
        return str(key) in self._cache

    def delete(self, key: CacheKey) -> bool:
        key_str = str(key)
        if key_str in self._cache:
            del self._cache[key_str]
            self._metadata.pop(key_str, None)
            return True
        return False

    def clear(self) -> None:
        """Limpa todo o cache."""
        self._cache.clear()
        self._metadata.clear()
        logger.info("In-memory cache cleared")

    def size(self) -> int:
        return len(self._cache)

    def memory_usage_mb(self) -> float:
        """Retorna uso de memória aproximado em MB."""
        total_bytes = sum(len(v) for v in self._cache.values())
        return total_bytes / (1024 * 1024)


# ============================================================================
# Disk Cache
# ============================================================================

class DiskCache(CacheBackend[bytes]):
    """
    Cache em filesystem para persistência de longo prazo.
    
    Características:
    - Persistente entre sessões
    - Preparado para compressão/serialização futura
    - Suporta metadados em JSON
    - Acesso O(1) com lookup em filesystem
    """

    # Sufixos padrão para diferentes tipos de dado
    _DATA_SUFFIX: Final[str] = "bin"
    _META_SUFFIX: Final[str] = "json"

    def __init__(self, base_dir: Path | str = ".cache/skymap"):
        """
        Inicializa cache em disco.
        
        Args:
            base_dir: Diretório base para armazenar cache
        """
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Disk cache initialized at {self._base_dir}")

    def _data_path(self, key: CacheKey) -> Path:
        """Caminho para arquivo de dados."""
        return self._base_dir / f"{key.key_hash}.{self._DATA_SUFFIX}"

    def _meta_path(self, key: CacheKey) -> Path:
        """Caminho para arquivo de metadados."""
        return self._base_dir / f"{key.key_hash}.{self._META_SUFFIX}"

    def get(self, key: CacheKey) -> bytes | None:
        """Lê dados do disco."""
        path = self._data_path(key)
        if not path.exists():
            return None

        try:
            return path.read_bytes()
        except OSError as e:
            logger.warning(f"Failed to read cache {key.key_hash[:8]}...: {e}")
            return None

    def set(self, key: CacheKey, value: bytes) -> None:
        """Escreve dados e metadados no disco."""
        try:
            # Escreve dados
            data_path = self._data_path(key)
            data_path.write_bytes(value)

            # Escreve metadados
            meta_path = self._meta_path(key)
            meta_json = json.dumps(
                {
                    "key": key.key_hash,
                    "metadata": key.metadata,
                    "size_bytes": len(value),
                }
            )
            meta_path.write_text(meta_json)

            logger.debug(f"Cached {key.key_hash[:8]}... ({len(value)} bytes)")
        except OSError as e:
            logger.error(f"Failed to write cache {key.key_hash[:8]}...: {e}")

    def exists(self, key: CacheKey) -> bool:
        return self._data_path(key).exists()

    def delete(self, key: CacheKey) -> bool:
        """Remove dados e metadados."""
        data_path = self._data_path(key)
        meta_path = self._meta_path(key)

        deleted = False
        for path in (data_path, meta_path):
            if path.exists():
                try:
                    path.unlink()
                    deleted = True
                except OSError as e:
                    logger.warning(f"Failed to delete {path}: {e}")

        return deleted

    def clear(self) -> None:
        """Remove todos os arquivos de cache."""
        import shutil

        try:
            shutil.rmtree(self._base_dir)
            self._base_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Disk cache cleared")
        except OSError as e:
            logger.error(f"Failed to clear cache: {e}")

    def size(self) -> int:
        """Retorna número de entradas em disco."""
        return len(list(self._base_dir.glob(f"*.{self._DATA_SUFFIX}")))

    def disk_usage_mb(self) -> float:
        """Retorna uso em disco aproximado em MB."""
        total_bytes = sum(p.stat().st_size for p in self._base_dir.glob(f"*.{self._DATA_SUFFIX}"))
        return total_bytes / (1024 * 1024)


# ============================================================================
# Hybrid Cache
# ============================================================================

class HybridCache(CacheBackend[bytes]):
    """
    Cache híbrido que combina memória + disco com fallback inteligente.
    
    Estratégia:
    1. Busca em cache de memória (rápido)
    2. Se não encontrar, busca em disco
    3. Se encontrar em disco, carrega em memória
    4. Ao salvar, escreve em ambos os níveis
    
    Benefícios:
    - Performance de memória para acessos repetidos
    - Persistência de disco entre sessões
    - Limite de memória com fallback para disco
    """

    def __init__(
        self,
        memory_max_size: int = 50,
        disk_dir: Path | str = ".cache/skymap",
    ):
        """
        Inicializa cache híbrido.
        
        Args:
            memory_max_size: Tamanho máximo de cache em memória
            disk_dir: Diretório para cache em disco
        """
        self._memory = InMemoryCache(max_size=memory_max_size)
        self._disk = DiskCache(base_dir=disk_dir)
        logger.info("Hybrid cache initialized (memory + disk)")

    def get(self, key: CacheKey) -> bytes | None:
        """
        Busca com fallback: memória → disco.
        
        Se encontrar em disco, carrega em memória.
        """
        # Tenta memória primeiro
        value = self._memory.get(key)
        if value is not None:
            logger.debug(f"Cache HIT (memory): {key.key_hash[:8]}...")
            return value

        # Fallback para disco
        value = self._disk.get(key)
        if value is not None:
            logger.debug(f"Cache HIT (disk): {key.key_hash[:8]}...")
            # Carrega em memória para próximos acessos
            self._memory.set(key, value)
            return value

        logger.debug(f"Cache MISS: {key.key_hash[:8]}...")
        return None

    def set(self, key: CacheKey, value: bytes) -> None:
        """Armazena em memória e disco simultaneamente."""
        self._memory.set(key, value)
        self._disk.set(key, value)

    def exists(self, key: CacheKey) -> bool:
        """Verifica em memória ou disco."""
        return self._memory.exists(key) or self._disk.exists(key)

    def delete(self, key: CacheKey) -> bool:
        """Remove de memória e disco."""
        mem_deleted = self._memory.delete(key)
        disk_deleted = self._disk.delete(key)
        return mem_deleted or disk_deleted

    def clear(self) -> None:
        """Limpa memória e disco."""
        self._memory.clear()
        self._disk.clear()

    def size(self) -> int:
        """Retorna número total de entradas (deduplica memória/disco)."""
        return len(set(list(self._memory._cache.keys()) + 
                      [p.stem for p in Path(self._disk._base_dir).glob(f"*.{self._disk._DATA_SUFFIX}")]))

    def stats(self) -> dict[str, Any]:
        """Retorna estatísticas de cache híbrido."""
        return {
            "memory_entries": self._memory.size(),
            "memory_mb": self._memory.memory_usage_mb(),
            "disk_entries": self._disk.size(),
            "disk_mb": self._disk.disk_usage_mb(),
        }
