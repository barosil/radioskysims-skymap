"""
Cache transparente para funções de SkyMap.

Fornece decorador e contexto para cache automático
sem acoplar ao código da função.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable, TypeVar, cast

from .cache_impl import HybridCache, SkyMapCacheKeyGenerator

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheContext:
    """
    Contexto global de cache para SkyMap.
    
    Gerencia instância singleton de cache híbrido,
    permitindo ativar/desativar cache globalmente.
    """

    _instance: HybridCache | None = None
    _enabled: bool = True

    @classmethod
    def get_cache(cls) -> HybridCache:
        """Retorna instância singleton de cache."""
        if cls._instance is None:
            cls._instance = HybridCache()
        return cls._instance

    @classmethod
    def set_cache(cls, cache: HybridCache) -> None:
        """Define cache customizado."""
        cls._instance = cache

    @classmethod
    def is_enabled(cls) -> bool:
        """Verifica se cache está ativo."""
        return cls._enabled

    @classmethod
    def enable(cls) -> None:
        """Ativa cache."""
        cls._enabled = True
        logger.info("Cache enabled")

    @classmethod
    def disable(cls) -> None:
        """Desativa cache."""
        cls._enabled = False
        logger.info("Cache disabled")

    @classmethod
    def clear(cls) -> None:
        """Limpa cache."""
        cls.get_cache().clear()
        logger.info("Cache cleared")

    @classmethod
    def stats(cls) -> dict[str, Any]:
        """Retorna estatísticas de cache."""
        cache = cls.get_cache()
        return cache.stats()


def cacheable(
    key_params: list[str] | None = None,
    prefix: str = "skymap",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorador para cache transparente de funções.
    
    Cria chave de cache a partir de parâmetros selecionados,
    armazena/recupera resultado automaticamente.
    
    Uso:
        @cacheable(key_params=["freq_ghz", "nside", "components"])
        def generate_skymap(freq_ghz: float, nside: int, components: list) -> SkyMap:
            ...
    
    Args:
        key_params: Lista de nomes de parâmetros a incluir na chave
        prefix: Prefixo para identificar tipo de cache (opcional)
        
    Returns:
        Função decorada com cache transparente
    """

    def decorator(func: Callable[..., bytes]) -> Callable[..., bytes]:
        sig = inspect.signature(func)
        key_gen = SkyMapCacheKeyGenerator()

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> bytes:
            # Se cache desativado, executa normalmente
            if not CacheContext.is_enabled():
                return func(*args, **kwargs)

            # Constrói cache key a partir de parâmetros selecionados
            cache_params = {"prefix": prefix, "func": func.__name__}

            # Usa key_params se fornecido, senão usa todos os kwargs
            params_to_use = key_params or list(sig.parameters.keys())

            try:
                # Extrai valores dos parâmetros nomeados
                for param_name in params_to_use:
                    if param_name in kwargs:
                        cache_params[param_name] = kwargs[param_name]
                    else:
                        # Tenta recuperar de args posicionais
                        try:
                            param_idx = list(sig.parameters.keys()).index(param_name)
                            if param_idx < len(args):
                                cache_params[param_name] = args[param_idx]
                        except (ValueError, IndexError):
                            pass

                cache_key = key_gen.generate(**cache_params)
                cache = CacheContext.get_cache()

                # Tenta recuperar do cache
                cached_value = cache.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache HIT: {func.__name__}")
                    return cached_value

                # Executa função
                result = func(*args, **kwargs)

                # Armazena em cache
                if isinstance(result, bytes):
                    cache.set(cache_key, result)
                    logger.debug(f"Cache SET: {func.__name__}")

                return result

            except Exception as e:
                logger.warning(f"Cache error in {func.__name__}: {e}")
                # Executa sem cache em caso de erro
                return func(*args, **kwargs)

        return wrapper

    return decorator


class CachedFunction:
    """
    Wrapper de função com cache gerenciável.
    
    Permite chamar função com/sem cache dinamicamente,
    sem precisar de decorador.
    
    Uso:
        def load_skymap(file_path: str) -> SkyMap:
            ...
        
        cached_load = CachedFunction(load_skymap, ["file_path"])
        
        # Com cache
        result = cached_load.call_cached(file_path="data.fits")
        
        # Sem cache
        result = cached_load.call_uncached(file_path="data.fits")
        
        # Limpar específico
        cached_load.clear(file_path="data.fits")
    """

    def __init__(
        self,
        func: Callable[..., Any],
        key_params: list[str] | None = None,
        prefix: str = "skymap",
    ):
        self.func = func
        self.key_params = key_params or list(inspect.signature(func).parameters.keys())
        self.prefix = prefix
        self.key_gen = SkyMapCacheKeyGenerator()
        self.sig = inspect.signature(func)

    def _make_key(self, **kwargs: Any) -> str:
        """Gera chave para dados fornecidos."""
        cache_params = {"prefix": self.prefix, "func": self.func.__name__}

        for param_name in self.key_params:
            if param_name in kwargs:
                cache_params[param_name] = kwargs[param_name]

        cache_key = self.key_gen.generate(**cache_params)
        return cache_key.key_hash

    def call_cached(self, *args: Any, **kwargs: Any) -> Any:
        """Executa com cache."""
        if not CacheContext.is_enabled():
            return self.func(*args, **kwargs)

        try:
            key = self._make_key(**kwargs)
            cache = CacheContext.get_cache()

            # Converte args para kwargs para chave de cache
            bound = self.sig.bind(*args, **kwargs)
            bound.apply_defaults()

            cache_key = self.key_gen.generate(
                prefix=self.prefix,
                func=self.func.__name__,
                **{k: v for k, v in bound.arguments.items() if k in self.key_params},
            )

            cached = cache.get(cache_key)
            if cached is not None:
                return cached

            result = self.func(*args, **kwargs)

            if isinstance(result, bytes):
                cache.set(cache_key, result)

            return result
        except Exception as e:
            logger.warning(f"Cache error: {e}")
            return self.func(*args, **kwargs)

    def call_uncached(self, *args: Any, **kwargs: Any) -> Any:
        """Executa sem cache."""
        return self.func(*args, **kwargs)

    def clear(self, **kwargs: Any) -> bool:
        """Remove chave específica do cache."""
        try:
            cache_key = self.key_gen.generate(
                prefix=self.prefix,
                func=self.func.__name__,
                **kwargs,
            )
            cache = CacheContext.get_cache()
            return cache.delete(cache_key)
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
            return False
