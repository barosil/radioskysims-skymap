# tests/domain/skymap/test_models.py
from __future__ import annotations

from unittest.mock import PropertyMock, patch

import healpy as hp
import numpy as np
import pytest

from radioskysims_skymap.domain.skymap.models import SkyMap
from radioskysims_skymap.domain.skymap.value_objects import NSide, SkyMapDataset
from radioskysims_skymap.domain.skymap.exceptions import InvalidDatasetError


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

NSIDE = NSide(8)


def make_map() -> np.ndarray:
    """Retorna um mapa HEALPix válido."""
    return np.ones(hp.nside2npix(NSIDE.value), dtype=np.float64)


# ---------------------------------------------------------------------
# Construção da entidade
# ---------------------------------------------------------------------


def test_create_skymap_from_numpy_array() -> None:
    """
    Deve criar um SkyMap diretamente de um numpy.ndarray.
    """

    data = make_map()

    skymap = SkyMap(
        dataset=data,
        nside=NSIDE,
        unit="K",
    )

    assert isinstance(skymap, SkyMap)
    assert skymap.nside == NSIDE
    assert skymap.nside_value == 8
    assert skymap.npix == hp.nside2npix(8)
    assert skymap.unit == "K"
    assert skymap.frame == "galactic"
    assert skymap.field == "I"


def test_create_skymap_from_dataset_value_object() -> None:
    """
    Deve aceitar SkyMapDataset como entrada.
    """

    dataset = SkyMapDataset(make_map())

    skymap = SkyMap(
        dataset=dataset,
        nside=NSIDE,
        unit="K",
    )

    assert isinstance(skymap, SkyMap)
    assert skymap.nside == NSIDE
    assert skymap.npix == dataset.size


def test_create_from_array_factory_method() -> None:
    """
    Deve criar um SkyMap através de SkyMap.from_array().
    """

    data = make_map()

    skymap = SkyMap.from_array(
        data=data,
        nside=NSIDE,
        unit="Jy",
        frame="equatorial",
        freq_ghz=1.42,
        field="I",
    )

    assert isinstance(skymap, SkyMap)
    assert skymap.unit == "Jy"
    assert skymap.frame == "equatorial"
    assert skymap.freq_ghz == pytest.approx(1.42)
    assert skymap.field == "I"


def test_create_from_alm_factory_method() -> None:
    """
    Deve criar um SkyMap a partir de coeficientes alm.
    """

    data = make_map()

    alm = hp.map2alm(data)

    skymap = SkyMap.from_alm(
        alm=alm,
        nside=NSIDE,
        unit="K",
    )

    assert isinstance(skymap, SkyMap)
    assert skymap.nside == NSIDE
    assert skymap.npix == hp.nside2npix(NSIDE.value)

    np.testing.assert_equal(
        skymap.data.shape,
        data.shape,
    )


# ---------------------------------------------------------------------
# Validações
# ---------------------------------------------------------------------


def test_invalid_dataset_type() -> None:
    """
    Deve falhar quando dataset possui tipo inválido.
    """

    with pytest.raises(TypeError):
        SkyMap(
            dataset="invalid",
            nside=NSIDE,
            unit="K",
        )


def test_dataset_must_be_1d() -> None:
    """
    Deve rejeitar datasets bidimensionais.
    """

    data = np.ones((10, 10))

    with pytest.raises(InvalidDatasetError):
        SkyMap(
            dataset=data,
            nside=NSIDE,
            unit="K",
        )


def test_empty_unit() -> None:
    """
    Unit deve ser uma string não vazia.
    """

    with pytest.raises(InvalidDatasetError):
        SkyMap(
            dataset=make_map(),
            nside=NSIDE,
            unit="",
        )


def test_invalid_frame() -> None:
    """
    Frame deve ser válido.
    """

    with pytest.raises(InvalidDatasetError):
        SkyMap(
            dataset=make_map(),
            nside=NSIDE,
            unit="K",
            frame="horizontal",
        )


def test_invalid_field() -> None:
    """
    Field deve ser I, Q ou U.
    """

    with pytest.raises(InvalidDatasetError):
        SkyMap(
            dataset=make_map(),
            nside=NSIDE,
            unit="K",
            field="V",
        )


def test_incompatible_dataset_size() -> None:
    """
    Dataset incompatível com o NSide deve falhar.
    """

    data = np.ones(100)

    with pytest.raises(InvalidDatasetError):
        SkyMap(
            dataset=data,
            nside=NSIDE,
            unit="K",
        )


def test_empty_dataset() -> None:
    """
    Dataset vazio deve ser rejeitado.
    """

    data = np.array([], dtype=np.float64)

    with pytest.raises(InvalidDatasetError):
        SkyMap(
            dataset=data,
            nside=NSIDE,
            unit="K",
        )


# ---------------------------------------------------------------------
# Propriedades
# ---------------------------------------------------------------------


def test_data_property_returns_numpy_array() -> None:
    """
    A propriedade data deve retornar um ndarray.
    """

    data = make_map()

    skymap = SkyMap(
        dataset=data,
        nside=NSIDE,
        unit="K",
    )

    result = skymap.data

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, data)


def test_data_property_returns_copy() -> None:
    """
    data deve retornar uma cópia dos dados internos.
    """

    data = make_map()

    skymap = SkyMap(
        dataset=data,
        nside=NSIDE,
        unit="K",
    )

    result = skymap.data
    result[:] = 0.0

    # O mapa interno não deve ser alterado
    assert np.all(skymap.data == 1.0)


def test_nside_property() -> None:
    """
    Deve retornar o Value Object NSide.
    """

    skymap = SkyMap(
        dataset=make_map(),
        nside=NSIDE,
        unit="K",
    )

    assert skymap.nside is NSIDE
    assert isinstance(skymap.nside, NSide)


def test_nside_value_property() -> None:
    """
    Deve retornar o valor inteiro do NSide.
    """

    skymap = SkyMap(
        dataset=make_map(),
        nside=NSIDE,
        unit="K",
    )

    assert skymap.nside_value == 8
    assert isinstance(skymap.nside_value, int)


def test_npix_property() -> None:
    """
    Deve retornar corretamente o número de pixels.
    """

    skymap = SkyMap(
        dataset=make_map(),
        nside=NSIDE,
        unit="K",
    )

    assert skymap.npix == hp.nside2npix(NSIDE.value)
    assert skymap.npix == NSIDE.npix


# ---------------------------------------------------------------------
# Métodos
# ---------------------------------------------------------------------


def test_copy_creates_new_instance() -> None:
    """
    copy() deve retornar uma nova instância de SkyMap.
    """

    original = SkyMap(
        dataset=make_map(),
        nside=NSIDE,
        unit="K",
        frame="galactic",
        freq_ghz=1.42,
        field="I",
    )

    copied = original.copy()

    assert copied is not original
    assert isinstance(copied, SkyMap)

    np.testing.assert_array_equal(
        copied.data,
        original.data,
    )

    assert copied.nside == original.nside
    assert copied.unit == original.unit
    assert copied.frame == original.frame
    assert copied.freq_ghz == original.freq_ghz
    assert copied.field == original.field


def test_copy_is_independent() -> None:
    """
    A cópia deve ser independente do objeto original.
    """

    original = SkyMap(
        dataset=make_map(),
        nside=NSIDE,
        unit="K",
    )

    copied = original.copy()

    modified = copied.data
    modified[:] = 0.0

    assert np.all(original.data == 1.0)


def test_apply_mask() -> None:
    """
    apply_mask deve aplicar corretamente a máscara.
    """

    data = make_map()

    mask = np.ones_like(data)
    mask[:10] = 0

    skymap = SkyMap(
        dataset=data,
        nside=NSIDE,
        unit="K",
    )

    masked = skymap.apply_mask(mask)

    assert isinstance(masked, SkyMap)

    np.testing.assert_array_equal(
        masked.data,
        data * mask,
    )


def test_apply_mask_invalid_type() -> None:
    """
    Máscara deve ser ndarray.
    """

    skymap = SkyMap(
        dataset=make_map(),
        nside=NSIDE,
        unit="K",
    )

    with pytest.raises(TypeError):
        skymap.apply_mask("mask")


def test_apply_mask_shape_mismatch() -> None:
    """
    Deve falhar quando o shape da máscara é diferente.
    """

    skymap = SkyMap(
        dataset=make_map(),
        nside=NSIDE,
        unit="K",
    )

    wrong_mask = np.ones(10)

    with pytest.raises(ValueError):
        skymap.apply_mask(wrong_mask)


def test_alm_returns_numpy_array() -> None:
    """
    alm() deve retornar coeficientes harmônicos.
    """

    skymap = SkyMap(
        dataset=make_map(),
        nside=NSIDE,
        unit="K",
    )

    alm = skymap.alm()

    assert isinstance(alm, np.ndarray)
    assert alm.size > 0


def test_alm_with_lmax() -> None:
    """
    alm() deve aceitar lmax.
    """

    skymap = SkyMap(
        dataset=make_map(),
        nside=NSIDE,
        unit="K",
    )

    alm = skymap.alm(lmax=8)

    assert isinstance(alm, np.ndarray)
    assert alm.size > 0


# ---------------------------------------------------------------------
# Validações adicionais
# ---------------------------------------------------------------------


def test_constructor_accepts_integer_nside() -> None:
    """
    Deve converter um NSide inteiro para o Value Object NSide.
    """
    skymap = SkyMap(
        dataset=make_map(),
        nside=8,
        unit="K",
    )

    assert isinstance(skymap.nside, NSide)
    assert skymap.nside == NSIDE
    assert skymap.nside_value == 8


@pytest.mark.parametrize(
    "invalid_unit",
    [
        None,
        1,
        1.0,
        object(),
    ],
)
def test_unit_must_be_string(
    invalid_unit: object,
) -> None:
    """
    Unit deve obrigatoriamente ser uma string.
    """
    with pytest.raises(
        TypeError,
        match="unit must be a string",
    ):
        SkyMap(
            dataset=make_map(),
            nside=NSIDE,
            unit=invalid_unit,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "empty_unit",
    [
        "",
        " ",
        "\t",
        "\n",
    ],
)
def test_unit_must_not_be_blank(
    empty_unit: str,
) -> None:
    """
    Unit não pode ser vazia nem conter somente espaços.
    """
    with pytest.raises(
        InvalidDatasetError,
        match="Unit cannot be empty",
    ):
        SkyMap(
            dataset=make_map(),
            nside=NSIDE,
            unit=empty_unit,
        )


@pytest.mark.parametrize(
    "invalid_frame",
    [
        None,
        1,
        1.0,
        object(),
    ],
)
def test_frame_must_be_string(
    invalid_frame: object,
) -> None:
    """
    Frame deve obrigatoriamente ser uma string.
    """
    with pytest.raises(
        TypeError,
        match="frame must be a string",
    ):
        SkyMap(
            dataset=make_map(),
            nside=NSIDE,
            unit="K",
            frame=invalid_frame,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "valid_frame",
    [
        "galactic",
        "equatorial",
        "ecliptic",
    ],
)
def test_accepts_all_supported_frames(
    valid_frame: str,
) -> None:
    """
    Deve aceitar todos os referenciais suportados.
    """
    skymap = SkyMap(
        dataset=make_map(),
        nside=NSIDE,
        unit="K",
        frame=valid_frame,
    )

    assert skymap.frame == valid_frame


@pytest.mark.parametrize(
    "invalid_field",
    [
        None,
        1,
        1.0,
        object(),
    ],
)
def test_field_must_be_string(
    invalid_field: object,
) -> None:
    """
    Field deve obrigatoriamente ser uma string.
    """
    with pytest.raises(
        TypeError,
        match="field must be a string",
    ):
        SkyMap(
            dataset=make_map(),
            nside=NSIDE,
            unit="K",
            field=invalid_field,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "valid_field",
    [
        "I",
        "Q",
        "U",
    ],
)
def test_accepts_all_supported_fields(
    valid_field: str,
) -> None:
    """
    Deve aceitar todos os campos de Stokes suportados.
    """
    skymap = SkyMap(
        dataset=make_map(),
        nside=NSIDE,
        unit="K",
        field=valid_field,
    )

    assert skymap.field == valid_field


# ---------------------------------------------------------------------
# Propriedades e métodos especiais
# ---------------------------------------------------------------------


def test_dataset_property_returns_value_object() -> None:
    """
    dataset deve retornar o Value Object armazenado pela entidade.
    """
    dataset = SkyMapDataset(make_map())

    skymap = SkyMap(
        dataset=dataset,
        nside=NSIDE,
        unit="K",
    )

    assert skymap.dataset is dataset
    assert isinstance(skymap.dataset, SkyMapDataset)


def test_len_returns_number_of_pixels() -> None:
    """
    len(skymap) deve retornar a quantidade de pixels.
    """
    skymap = SkyMap(
        dataset=make_map(),
        nside=NSIDE,
        unit="K",
    )

    assert len(skymap) == NSIDE.npix
    assert len(skymap) == skymap.npix


def test_repr_contains_main_metadata() -> None:
    """
    repr deve apresentar os principais metadados do mapa.
    """
    skymap = SkyMap(
        dataset=make_map(),
        nside=NSIDE,
        unit="K",
        frame="equatorial",
        field="Q",
    )

    result = repr(skymap)

    assert result == (
        f"SkyMap(nside=8, npix={NSIDE.npix}, unit='K', frame='equatorial', field='Q')"
    )


# ---------------------------------------------------------------------
# Igualdade e hash
# ---------------------------------------------------------------------


def test_equal_skymaps() -> None:
    """
    Mapas com dados e metadados iguais devem ser equivalentes.
    """
    first = SkyMap(
        dataset=make_map(),
        nside=NSIDE,
        unit="K",
        frame="galactic",
        freq_ghz=100.0,
        field="I",
    )

    second = SkyMap(
        dataset=make_map(),
        nside=NSIDE,
        unit="K",
        frame="galactic",
        freq_ghz=100.0,
        field="I",
    )

    assert first == second
    assert second == first


def test_skymap_is_not_equal_to_other_type() -> None:
    """
    SkyMap não deve ser igual a objetos de outro tipo.
    """
    skymap = SkyMap(
        dataset=make_map(),
        nside=NSIDE,
        unit="K",
    )

    assert skymap != object()
    assert skymap != make_map()


@pytest.mark.parametrize(
    (
        "unit",
        "frame",
        "freq_ghz",
        "field",
    ),
    [
        ("Jy", "galactic", 100.0, "I"),
        ("K", "equatorial", 100.0, "I"),
        ("K", "galactic", 143.0, "I"),
        ("K", "galactic", 100.0, "Q"),
    ],
)
def test_skymaps_with_different_metadata_are_not_equal(
    unit: str,
    frame: str,
    freq_ghz: float,
    field: str,
) -> None:
    """
    Diferenças de metadados devem tornar os mapas distintos.
    """
    reference = SkyMap(
        dataset=make_map(),
        nside=NSIDE,
        unit="K",
        frame="galactic",
        freq_ghz=100.0,
        field="I",
    )

    different = SkyMap(
        dataset=make_map(),
        nside=NSIDE,
        unit=unit,
        frame=frame,
        freq_ghz=freq_ghz,
        field=field,
    )

    assert reference != different


def test_skymaps_with_different_data_are_not_equal() -> None:
    """
    Diferenças nos dados devem tornar os mapas distintos.
    """
    first_data = make_map()
    second_data = make_map()
    second_data[0] = 2.0

    first = SkyMap(
        dataset=first_data,
        nside=NSIDE,
        unit="K",
    )

    second = SkyMap(
        dataset=second_data,
        nside=NSIDE,
        unit="K",
    )

    assert first != second


def test_equal_skymaps_have_equal_hashes() -> None:
    """
    Mapas equivalentes devem possuir o mesmo hash.
    """
    first = SkyMap(
        dataset=make_map(),
        nside=NSIDE,
        unit="K",
        frame="galactic",
        freq_ghz=100.0,
        field="I",
    )

    second = SkyMap(
        dataset=make_map(),
        nside=NSIDE,
        unit="K",
        frame="galactic",
        freq_ghz=100.0,
        field="I",
    )

    assert hash(first) == hash(second)


def test_hash_changes_with_metadata() -> None:
    """
    Metadados diferentes devem participar da composição do hash.
    """
    first = SkyMap(
        dataset=make_map(),
        nside=NSIDE,
        unit="K",
        freq_ghz=100.0,
    )

    second = SkyMap(
        dataset=make_map(),
        nside=NSIDE,
        unit="K",
        freq_ghz=143.0,
    )

    assert hash(first) != hash(second)


# ---------------------------------------------------------------------
# Operações adicionais
# ---------------------------------------------------------------------


def test_apply_mask_preserves_metadata() -> None:
    """
    apply_mask deve preservar os metadados do mapa original.
    """
    skymap = SkyMap(
        dataset=make_map(),
        nside=NSIDE,
        unit="K",
        frame="equatorial",
        freq_ghz=1.42,
        field="Q",
    )

    masked = skymap.apply_mask(np.ones(NSIDE.npix, dtype=np.float64))

    assert masked is not skymap
    assert masked.nside == skymap.nside
    assert masked.unit == skymap.unit
    assert masked.frame == skymap.frame
    assert masked.freq_ghz == skymap.freq_ghz
    assert masked.field == skymap.field


# ---------------------------------------------------------------------
# Integração com Healpy
# ---------------------------------------------------------------------


@patch("radioskysims_skymap.domain.skymap.models.hp.map2alm")
def test_alm_delegates_to_healpy(
    mock_map2alm,
) -> None:
    """
    alm deve encaminhar os dados e lmax ao Healpy.
    """
    expected = np.array(
        [1.0 + 0.0j, 2.0 + 1.0j],
        dtype=np.complex128,
    )

    mock_map2alm.return_value = expected

    skymap = SkyMap(
        dataset=make_map(),
        nside=NSIDE,
        unit="K",
    )

    result = skymap.alm(lmax=16)

    mock_map2alm.assert_called_once()

    args, kwargs = mock_map2alm.call_args

    np.testing.assert_array_equal(
        args[0],
        skymap.data,
    )

    assert kwargs == {"lmax": 16}

    np.testing.assert_array_equal(
        result,
        expected,
    )


@patch("radioskysims_skymap.domain.skymap.models.hp.map2alm")
def test_alm_converts_result_to_complex128(
    mock_map2alm,
) -> None:
    """
    alm deve normalizar o resultado para complex128.
    """
    mock_map2alm.return_value = np.array(
        [1.0 + 0.0j],
        dtype=np.complex64,
    )

    skymap = SkyMap(
        dataset=make_map(),
        nside=NSIDE,
        unit="K",
    )

    result = skymap.alm()

    assert result.dtype == np.complex128


@patch("radioskysims_skymap.domain.skymap.models.hp.alm2map")
def test_from_alm_delegates_to_healpy(
    mock_alm2map,
) -> None:
    """
    from_alm deve encaminhar os coeficientes e NSide ao Healpy.
    """
    alm = np.array(
        [1.0 + 0.0j, 2.0 + 0.0j],
        dtype=np.complex128,
    )

    generated_data = make_map().astype(np.float32)

    mock_alm2map.return_value = generated_data

    skymap = SkyMap.from_alm(
        alm=alm,
        nside=NSIDE,
        unit="K",
        frame="ecliptic",
        freq_ghz=30.0,
        field="U",
    )

    mock_alm2map.assert_called_once_with(
        alm,
        NSIDE.value,
    )

    assert skymap.data.dtype == np.float64
    assert skymap.frame == "ecliptic"
    assert skymap.freq_ghz == 30.0
    assert skymap.field == "U"

    np.testing.assert_array_equal(
        skymap.data,
        generated_data.astype(np.float64),
    )


def test_npix_falls_back_to_nside_when_dataset_size_is_none() -> None:
    """
    npix deve usar NSide.npix quando o tamanho do dataset
    não estiver disponível.
    """
    dataset = SkyMapDataset(make_map())

    with patch.object(
        SkyMapDataset,
        "size",
        new_callable=PropertyMock,
        return_value=None,
    ):
        skymap = SkyMap(
            dataset=dataset,
            nside=NSIDE,
            unit="K",
        )

        assert skymap.npix == NSIDE.npix
        assert len(skymap) == NSIDE.npix
