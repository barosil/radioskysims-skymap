from unittest.mock import ANY, patch

import numpy as np
from radioskysims_skymap.domain.skymap.operations import SkyMapOperations


@patch("radioskysims_skymap.domain.skymap.operations.hp")
def test_rotate_success(mock_hp) -> None:
    """Verifica a sequência de chamadas do Healpy."""

    map_data = np.ones(12, dtype=np.float64)

    alm = np.array(
        [1.0 + 0.0j, 2.0 + 0.0j],
        dtype=np.complex128,
    )

    expected = np.ones(12, dtype=np.float64)

    mock_hp.get_nside.return_value = 1
    mock_hp.map2alm.return_value = alm
    mock_hp.alm2map.return_value = expected

    result = SkyMapOperations.rotate(
        map_data,
        (0.0, 0.0, 0.0),
    )

    mock_hp.get_nside.assert_called_once_with(map_data)
    mock_hp.map2alm.assert_called_once_with(map_data)

    mock_hp.rotate_alm.assert_called_once_with(
        ANY,
        psi=0.0,
        theta=0.0,
        phi=0.0,
    )

    mock_hp.alm2map.assert_called_once_with(
        ANY,
        nside=1,
    )

    rotated_alm = mock_hp.rotate_alm.call_args.args[0]

    assert rotated_alm is not alm

    np.testing.assert_array_equal(
        rotated_alm,
        alm,
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )
