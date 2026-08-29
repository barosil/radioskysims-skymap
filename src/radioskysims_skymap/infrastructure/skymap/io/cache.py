# src/radioskysims/infrastructure/skymap/io/cache.py

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CacheStore:
    """
    Cache simples baseado em filesystem (API básica e backwards-compatible).
    
    Interface minimalista para operações de cache direto:
    - save(payload, suffix, data) → Path
    - load(payload, suffix) → Path
    - exists(payload, suffix) → bool
    - clear() → None
    
    Para cache mais avançado (memória, híbrido, contexto global),
    use as implementações em cache_impl.py e decorador cacheable.
    
    Exemplo:
        store = CacheStore(Path(".cache"))
        path = store.save({"freq": 30.0, "nside": 1}, "bin", data_bytes)
        if store.exists({"freq": 30.0, "nside": 1}, "bin"):
            cached_path = store.load({"freq": 30.0, "nside": 1}, "bin")
    """

    def __init__(self, base_dir: Path | str = ".cache/skymap"):
        """
        Inicializa cache em filesystem.
        
        Args:
            base_dir: Diretório base para armazenar cache
        """
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"CacheStore initialized at {self._base_dir}")

    # --------------------------------------------------
    # Chaves
    # --------------------------------------------------
    def _make_key(self, payload: dict[str, Any]) -> str:
        """
        Gera chave hash determinística a partir de um payload.
        
        Usa JSON com sorted_keys para garantir ordem determinística.
        
        Args:
            payload: Dicionário com parâmetros
            
        Returns:
            Hash SHA256 hexadecimal
        """
        serialized = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(serialized).hexdigest()

    def _path_for(self, key: str, suffix: str) -> Path:
        """Constrói caminho para arquivo de cache."""
        return self._base_dir / f"{key}.{suffix}"

    # --------------------------------------------------
    # Interface pública
    # --------------------------------------------------
    def exists(self, payload: dict[str, Any], suffix: str) -> bool:
        """
        Verifica se entrada está em cache.
        
        Args:
            payload: Parâmetros (dicionário)
            suffix: Sufixo de arquivo (ex: "bin", "fits", "json")
            
        Returns:
            True se arquivo existe, False caso contrário
        """
        key = self._make_key(payload)
        return self._path_for(key, suffix).exists()

    def load(self, payload: dict[str, Any], suffix: str) -> Path:
        """
        Carrega caminho de arquivo em cache.
        
        Retorna Path para arquivo existente, pode ser lido diretamente.
        
        Args:
            payload: Parâmetros (dicionário)
            suffix: Sufixo de arquivo
            
        Returns:
            Path existente para arquivo de cache
            
        Raises:
            FileNotFoundError: Se cache não existe
        """
        key = self._make_key(payload)
        path = self._path_for(key, suffix)

        if not path.exists():
            logger.debug(f"Cache miss: {key[:8]}...")
            raise FileNotFoundError(f"Cached file not found: {path}")

        logger.debug(f"Cache hit: {key[:8]}...")
        return path

    def save(
        self,
        payload: dict[str, Any],
        suffix: str,
        data: bytes,
    ) -> Path:
        """
        Salva dados em cache.
        
        Args:
            payload: Parâmetros (dicionário) para gerar chave
            suffix: Sufixo de arquivo
            data: Dados em bytes
            
        Returns:
            Path onde arquivo foi salvo
        """
        key = self._make_key(payload)
        path = self._path_for(key, suffix)

        try:
            path.write_bytes(data)
            logger.debug(f"Cache saved: {key[:8]}... ({len(data)} bytes)")
            return path
        except OSError as e:
            logger.error(f"Failed to save cache: {e}")
            raise

    def clear(self) -> None:
        """
        Remove todos os arquivos de cache.
        
        Limpa diretório completamente.
        """
        import shutil

        try:
            shutil.rmtree(self._base_dir)
            self._base_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cache store cleared")
        except OSError as e:
            logger.error(f"Failed to clear cache: {e}")

    def size(self) -> int:
        """
        Retorna número de arquivos em cache.
        
        Returns:
            Contagem de arquivos no diretório de cache
        """
        return len(list(self._base_dir.glob("*")))

    def disk_usage_mb(self) -> float:
        """
        Retorna uso em disco aproximado.
        
        Returns:
            Tamanho total em MB
        """
        total_bytes = sum(p.stat().st_size for p in self._base_dir.glob("*"))
        return total_bytes / (1024 * 1024)
