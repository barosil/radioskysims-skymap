"""
Cache infrastructure para SkyMap.

Fornece três níveis de API:

1. **Básico (CacheStore)**:
   - Interface simples get/set/exists
   - Para uso direto em operações
   
2. **Intermediário (InMemoryCache, DiskCache, HybridCache)**:
   - Implementações específicas de cache
   - Customizáveis e compostas
   
3. **Avançado (cacheable, CacheContext)**:
   - Cache transparente com decoradores
   - Gerenciamento global de cache
   
Exemplo de uso:

    # 1. Cache básico (backwards compatible)
    store = CacheStore(Path(".cache"))
    if store.exists({"freq": 30.0, "nside": 1}, "bin"):
        path = store.load({"freq": 30.0, "nside": 1}, "bin")
    
    # 2. Cache intermediário
    cache = HybridCache()
    key = SkyMapCacheKeyGenerator().generate(freq_ghz=30.0, nside=1)
    
    cache.set(key, data_bytes)
    retrieved = cache.get(key)
    
    # 3. Cache avançado (decorador)
    @cacheable(key_params=["freq_ghz", "nside", "components"])
    def generate_skymap(freq_ghz: float, nside: int, components: list) -> bytes:
        ...
    
    # Gerenciamento global
    CacheContext.enable()
    CacheContext.disable()
    stats = CacheContext.stats()
    CacheContext.clear()
"""

from .cache import CacheStore
from .cache_base import CacheBackend, CacheEntry, CacheKey, CacheKeyGenerator
from .cache_context import CacheContext, CachedFunction, cacheable
from .cache_impl import DiskCache, HybridCache, InMemoryCache, SkyMapCacheKeyGenerator

__all__ = [
    # Básico (backwards compatible)
    "CacheStore",
    # Base abstratas
    "CacheBackend",
    "CacheEntry",
    "CacheKey",
    "CacheKeyGenerator",
    # Implementações
    "InMemoryCache",
    "DiskCache",
    "HybridCache",
    "SkyMapCacheKeyGenerator",
    # Cache transparente
    "cacheable",
    "CacheContext",
    "CachedFunction",
]
