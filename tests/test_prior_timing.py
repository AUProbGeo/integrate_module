"""Computation times stored as attributes on prior model/data arrays
(``compute_time``, ``compute_call``, ``compute_date``), see
``integrate_io.save_compute_time`` and ``integrate_io.get_prior_timing``."""
import h5py
import numpy as np
import pytest

import integrate as ig


def _attrs(f_h5, key):
    with h5py.File(f_h5, 'r') as f:
        return dict(f[key].attrs)


@pytest.fixture
def prior_layered(tmp_path):
    f = str(tmp_path / 'prior_layered.h5')
    out = ig.prior_model_layered(N=500, NLAY_min=2, NLAY_max=4, f_prior_h5=f, showInfo=-1)
    return out


def test_prior_model_generator_stamps_all_models(prior_layered):
    with h5py.File(prior_layered, 'r') as f:
        keys = ['/' + k for k in f.keys() if k[0] == 'M']
    assert len(keys) >= 2
    a = [_attrs(prior_layered, k) for k in keys]
    for ai in a:
        assert ai['compute_time'] > 0
        assert str(ai['compute_call']).startswith('prior_model_layered@')
    # one call -> one shared time and call id
    assert len({ai['compute_call'] for ai in a}) == 1
    assert len({ai['compute_time'] for ai in a}) == 1


def test_identity_data_copy_and_totals(prior_layered, tmp_path):
    f_data = str(tmp_path / 'prior_data.h5')
    f_out, id_ = ig.prior_data_identity(prior_layered, im=1, id=1, doMakePriorCopy=True,
                                        f_prior_data_h5=f_data, showInfo=-1)
    a = _attrs(f_out, '/D%d' % id_)
    assert a['compute_time'] >= 0
    assert str(a['compute_call']).startswith('prior_data_identity@')
    # model timing survived the copy into the prior-data file
    assert 'compute_time' in _attrs(f_out, '/M1')

    t = ig.get_prior_timing(f_out, showInfo=0)
    t_model = _attrs(f_out, '/M1')['compute_time']
    assert t['T_model'] == pytest.approx(t_model)   # shared call counted once
    assert t['T_data'] == pytest.approx(a['compute_time'])
    assert t['T_total'] == pytest.approx(t['T_model'] + t['T_data'])


def test_save_compute_time_and_old_file(tmp_path):
    f = str(tmp_path / 'external.h5')
    ig.save_prior_model(f, np.random.rand(100, 5), im=1, showInfo=-1)
    ig.save_prior_data(f, np.random.rand(100, 3), id=1, showInfo=-1)
    # no timing yet (like an old file, or a prior written by another program)
    t = ig.get_prior_timing(f, showInfo=0)
    assert t['datasets']['/M1']['compute_time'] is None
    assert t['T_total'] == 0.0
    # stamp it by hand, e.g. after geoprior1d
    ig.save_compute_time(f, ['/M1'], 12.5, 'geoprior1d')
    ig.save_compute_time(f, '/D1', 3.0, 'my_forward')
    t = ig.get_prior_timing(f, showInfo=0)
    assert t['T_model'] == pytest.approx(12.5)
    assert t['T_data'] == pytest.approx(3.0)


def test_borehole_prior_data_is_stamped(tmp_path):
    f = str(tmp_path / 'litho.h5')
    rng = np.random.default_rng(0)
    z = np.arange(0, 30, 1.0)
    M = rng.integers(1, 4, size=(300, len(z)))
    ig.save_prior_model(f, M, im=1, x=z, is_discrete=1, class_id=np.array([1, 2, 3]),
                        class_name=np.array([b'clay', b'sand', b'gravel']), showInfo=-1)
    BH = {'name': 'BH1', 'X': 0.0, 'Y': 0.0, 'method': 'mode_probability',
          'depth_top': np.array([0.0, 10.0]), 'depth_bottom': np.array([10.0, 20.0]),
          'class_obs': np.array([1, 2]), 'class_prob': np.array([0.8, 0.8])}
    _, id_prior = ig.prior_data_borehole(f, 1, BH, showInfo=0)
    a = _attrs(f, '/D%d' % id_prior)
    assert a['compute_time'] >= 0
    assert str(a['compute_call']).startswith('prior_data_borehole[BH1]@')
