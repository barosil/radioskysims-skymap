# radioskysims-skymap

SkyMap do ecossistema **RadioSkySim**.

## Conteúdo

- `domain.skymap` — `SkyMap`, `NSide`, `SkyMapDataset`, mask, operations, analysis
- `application.skymap` — use cases de análise/geração (`analyze`, `generate`, `dto`, `ports`)
- `application.ports.skymap` + `skymap_repository` — contratos
- `infrastructure.skymap` — providers (healpy/pysm/HIPS), adapters, repositório
  HDF5, IO/cache, telemetria
- `infrastructure.db.skymap_models` — modelos SQLModel

**Não inclui**: `generate_continuum` (usa radiosources → `radioskysims-sources`)
e `tod_projector` (integração TOD → `radioskysims-simulation`).

## Dependências

`numpy`, `astropy`, `xarray`, `healpy`, `h5py`, `pysm3`, `radioskysims-core`.
**Sem** `astroquery`, `pyccl`, `camb`, `dask`, `panel`.

## Testes

```bash
uv run --package radioskysims-skymap pytest
```
