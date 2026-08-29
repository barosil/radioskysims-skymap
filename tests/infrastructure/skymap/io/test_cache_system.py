"""
Testes para sistema de cache de SkyMap.

Cobre:
1. Cache básico (CacheStore) - backwards compatible
2. Cache em memória (InMemoryCache) - performance
3. Cache em disco (DiskCache) - persistência
4. Cache híbrido (HybridCache) - memória + disco
5. Cache transparente (cacheable, CacheContext) - uso fácil
"""

import tempfile
from pathlib import Path

import pytest
import numpy as np

from radioskysims_skymap.infrastructure.skymap.io.cache import CacheStore
from radioskysims_skymap.infrastructure.skymap.io.cache_impl import (
    InMemoryCache,
    DiskCache,
    HybridCache,
    SkyMapCacheKeyGenerator,
)
from radioskysims_skymap.infrastructure.skymap.io.cache_context import (
    CacheContext,
    CachedFunction,
    cacheable,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_data() -> bytes:
    """Dados de amostra para testes."""
    return np.ones(12, dtype=np.float32).tobytes()


@pytest.fixture
def key_generator() -> SkyMapCacheKeyGenerator:
    """Gerador de chaves."""
    return SkyMapCacheKeyGenerator()


@pytest.fixture
def temp_cache_dir() -> Path:
    """Diretório temporário para cache em disco."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Cache Básico (CacheStore)
# ============================================================================

class TestCacheStore:
    """Testes para CacheStore (API básica)."""

    def test_cache_store_save_and_load(self, temp_cache_dir: Path, sample_data: bytes):
        """Testa save + load básico."""
        store = CacheStore(temp_cache_dir)

        payload = {"freq_ghz": 30.0, "nside": 1}
        path = store.save(payload, "bin", sample_data)

        assert path.exists()
        assert path.read_bytes() == sample_data

    def test_cache_store_exists(self, temp_cache_dir: Path, sample_data: bytes):
        """Testa verificação de existência."""
        store = CacheStore(temp_cache_dir)
        payload = {"freq_ghz": 30.0, "nside": 1}

        assert not store.exists(payload, "bin")
        store.save(payload, "bin", sample_data)
        assert store.exists(payload, "bin")

    def test_cache_store_deterministic_key(self, temp_cache_dir: Path, sample_data: bytes):
        """Verifica que chave é determinística."""
        store = CacheStore(temp_cache_dir)

        payload1 = {"freq_ghz": 30.0, "nside": 1, "components": ["synchrotron"]}
        payload2 = {"components": ["synchrotron"], "freq_ghz": 30.0, "nside": 1}

        path1 = store.save(payload1, "bin", sample_data)
        path2 = store.load(payload2, "bin")

        assert path1 == path2  # Mesma chave apesar de ordem diferente

    def test_cache_store_clear(self, temp_cache_dir: Path, sample_data: bytes):
        """Testa limpeza de cache."""
        store = CacheStore(temp_cache_dir)

        store.save({"freq_ghz": 30.0}, "bin", sample_data)
        store.save({"freq_ghz": 60.0}, "bin", sample_data)

        assert store.size() == 2
        store.clear()
        assert store.size() == 0


# ============================================================================
# Cache em Memória
# ============================================================================

class TestInMemoryCache:
    """Testes para cache em memória."""

    def test_in_memory_get_set(self, key_generator: SkyMapCacheKeyGenerator, sample_data: bytes):
        """Testa armazenamento e recuperação."""
        cache = InMemoryCache()
        key = key_generator.generate(freq_ghz=30.0, nside=1)

        cache.set(key, sample_data)
        assert cache.get(key) == sample_data

    def test_in_memory_exists(self, key_generator: SkyMapCacheKeyGenerator):
        """Testa verificação de existência."""
        cache = InMemoryCache()
        key = key_generator.generate(freq_ghz=30.0)

        assert not cache.exists(key)
        cache.set(key, b"test")
        assert cache.exists(key)

    def test_in_memory_lru_eviction(self, key_generator: SkyMapCacheKeyGenerator):
        """Testa política LRU de descartes."""
        cache = InMemoryCache(max_size=3)

        keys = [key_generator.generate(freq_ghz=float(i)) for i in range(5)]

        # Preenche cache
        for i, key in enumerate(keys[:3]):
            cache.set(key, f"data_{i}".encode())

        assert cache.size() == 3

        # Adiciona 4º item (deve descartar 1º - LRU)
        cache.set(keys[3], b"data_3")
        assert cache.size() == 3
        assert cache.get(keys[0]) is None  # Primeira foi descartada
        assert cache.get(keys[3]) == b"data_3"

    def test_in_memory_delete(self, key_generator: SkyMapCacheKeyGenerator):
        """Testa remoção manual."""
        cache = InMemoryCache()
        key = key_generator.generate(freq_ghz=30.0)

        cache.set(key, b"test")
        assert cache.delete(key)
        assert not cache.exists(key)
        assert not cache.delete(key)  # Já foi removido

    def test_in_memory_clear(self, key_generator: SkyMapCacheKeyGenerator):
        """Testa limpeza total."""
        cache = InMemoryCache()

        for i in range(5):
            key = key_generator.generate(freq_ghz=float(i))
            cache.set(key, f"data_{i}".encode())

        assert cache.size() == 5
        cache.clear()
        assert cache.size() == 0


# ============================================================================
# Cache em Disco
# ============================================================================

class TestDiskCache:
    """Testes para cache em disco."""

    def test_disk_cache_get_set(self, temp_cache_dir: Path, key_generator: SkyMapCacheKeyGenerator, sample_data: bytes):
        """Testa armazenamento em disco."""
        cache = DiskCache(temp_cache_dir)
        key = key_generator.generate(freq_ghz=30.0, nside=1)

        cache.set(key, sample_data)
        assert cache.exists(key)

        retrieved = cache.get(key)
        assert retrieved == sample_data

    def test_disk_cache_metadata(self, temp_cache_dir: Path, key_generator: SkyMapCacheKeyGenerator):
        """Testa armazenamento de metadados."""
        cache = DiskCache(temp_cache_dir)
        key = key_generator.generate(freq_ghz=30.0, nside=1, components=["synchrotron"])

        cache.set(key, b"test_data")

        # Verifica que arquivo de metadados foi criado
        meta_files = list(temp_cache_dir.glob("*.json"))
        assert len(meta_files) > 0

    def test_disk_cache_persistence(self, temp_cache_dir: Path, key_generator: SkyMapCacheKeyGenerator, sample_data: bytes):
        """Testa persistência entre instâncias."""
        key = key_generator.generate(freq_ghz=30.0)

        # Primeira instância
        cache1 = DiskCache(temp_cache_dir)
        cache1.set(key, sample_data)

        # Segunda instância (simula novo programa)
        cache2 = DiskCache(temp_cache_dir)
        assert cache2.exists(key)
        assert cache2.get(key) == sample_data

    def test_disk_cache_delete(self, temp_cache_dir: Path, key_generator: SkyMapCacheKeyGenerator):
        """Testa remoção de arquivo."""
        cache = DiskCache(temp_cache_dir)
        key = key_generator.generate(freq_ghz=30.0)

        cache.set(key, b"test")
        assert cache.delete(key)
        assert not cache.exists(key)


# ============================================================================
# Cache Híbrido
# ============================================================================

class TestHybridCache:
    """Testes para cache híbrido (memória + disco)."""

    def test_hybrid_cache_memory_first(self, temp_cache_dir: Path, key_generator: SkyMapCacheKeyGenerator, sample_data: bytes):
        """Testa que memória é consultada primeiro."""
        cache = HybridCache(disk_dir=temp_cache_dir)
        key = key_generator.generate(freq_ghz=30.0)

        cache.set(key, sample_data)

        # Deve recuperar de memória
        retrieved = cache.get(key)
        assert retrieved == sample_data
        assert cache._memory.exists(key)

    def test_hybrid_cache_disk_fallback(self, temp_cache_dir: Path, key_generator: SkyMapCacheKeyGenerator, sample_data: bytes):
        """Testa fallback para disco."""
        cache = HybridCache(disk_dir=temp_cache_dir)
        key = key_generator.generate(freq_ghz=30.0)

        cache.set(key, sample_data)

        # Simula perda de memória
        cache._memory.clear()
        assert not cache._memory.exists(key)
        assert cache._disk.exists(key)

        # Deve recuperar de disco
        retrieved = cache.get(key)
        assert retrieved == sample_data

        # E recarregar em memória
        assert cache._memory.exists(key)

    def test_hybrid_cache_stats(self, temp_cache_dir: Path, key_generator: SkyMapCacheKeyGenerator):
        """Testa coleta de estatísticas."""
        cache = HybridCache(disk_dir=temp_cache_dir)

        for i in range(3):
            key = key_generator.generate(freq_ghz=float(i))
            cache.set(key, b"x" * 1024)  # 1KB cada

        stats = cache.stats()
        assert stats["memory_entries"] == 3
        assert stats["disk_entries"] == 3


# ============================================================================
# Cache Transparente (Decorador)
# ============================================================================

class TestCacheableDecorator:
    """Testes para decorador @cacheable."""

    def test_cacheable_basic(self):
        """Testa cache com decorador."""
        call_count = 0

        @cacheable(key_params=["x"])
        def expensive_function(x: int) -> bytes:
            nonlocal call_count
            call_count += 1
            return f"result_{x}".encode()

        # Primeira chamada
        result1 = expensive_function(x=42)
        assert call_count == 1

        # Segunda chamada (deve vir de cache)
        result2 = expensive_function(x=42)
        assert call_count == 1  # Não incrementou
        assert result1 == result2

    def test_cacheable_disabled(self):
        """Testa com cache desativado."""
        call_count = 0

        @cacheable(key_params=["x"])
        def expensive_function(x: int) -> bytes:
            nonlocal call_count
            call_count += 1
            return f"result_{x}".encode()

        CacheContext.disable()
        try:
            expensive_function(x=42)
            expensive_function(x=42)
            assert call_count == 2  # Cache não funciona
        finally:
            CacheContext.enable()


class TestCacheContext:
    """Testes para contexto global de cache."""

    def test_cache_context_singleton(self):
        """Testa que CacheContext é singleton."""
        cache1 = CacheContext.get_cache()
        cache2 = CacheContext.get_cache()

        assert cache1 is cache2

    def test_cache_context_enable_disable(self):
        """Testa ativar/desativar cache."""
        assert CacheContext.is_enabled()

        CacheContext.disable()
        assert not CacheContext.is_enabled()

        CacheContext.enable()
        assert CacheContext.is_enabled()

    def test_cache_context_stats(self):
        """Testa coleta de estatísticas."""
        CacheContext.clear()

        key_gen = SkyMapCacheKeyGenerator()
        cache = CacheContext.get_cache()

        for i in range(3):
            key = key_gen.generate(freq_ghz=float(i))
            cache.set(key, b"x" * 1024)

        stats = CacheContext.stats()
        assert stats["memory_entries"] == 3
        assert stats["disk_entries"] == 3


class TestCachedFunction:
    """Testes para wrapper CachedFunction."""

    def test_cached_function_call_cached(self):
        """Testa execução com cache."""
        call_count = 0

        def my_function(x: int, y: int) -> bytes:
            nonlocal call_count
            call_count += 1
            return f"{x}+{y}".encode()

        cached_fn = CachedFunction(my_function, key_params=["x", "y"])

        # Primeira chamada
        result1 = cached_fn.call_cached(x=1, y=2)
        assert call_count == 1

        # Segunda chamada
        result2 = cached_fn.call_cached(x=1, y=2)
        assert call_count == 1  # Cache funciona
        assert result1 == result2

    def test_cached_function_call_uncached(self):
        """Testa execução sem cache."""
        call_count = 0

        def my_function(x: int) -> bytes:
            nonlocal call_count
            call_count += 1
            return f"result_{x}".encode()

        cached_fn = CachedFunction(my_function, key_params=["x"])

        cached_fn.call_uncached(x=42)
        cached_fn.call_uncached(x=42)

        assert call_count == 2  # Não usa cache

    def test_cached_function_clear(self):
        """Testa limpeza de entrada específica."""
        def my_function(x: int) -> bytes:
            return f"result_{x}".encode()

        cached_fn = CachedFunction(my_function, key_params=["x"])

        cached_fn.call_cached(x=42)
        assert cached_fn.clear(x=42)
        assert not cached_fn.clear(x=42)  # Já foi removido
