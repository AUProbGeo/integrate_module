"""Tests that the anemone forward backend works with anemone-core from PyPI.

anemone is installed as the ``anemone-core`` package (``pip install
anemone-core``, imported as ``anemone``).  These tests check

* that ``import anemone`` resolves to the installed ``anemone-core``
  distribution, version >= 0.1.1 (not a stray local checkout on the path);
* the two bugs in anemone-core 0.1.0 that 0.1.1 fixes:
  - ``FilterChain.stepoff_response`` with repeated cutoff frequencies (the
    default ``rx_coil_filter='two_pole'`` gives two identical poles), and
  - CUDA device handling in ``Forward``;
* that ``forward_anemone`` / ``prior_data_anemone`` run end to end and agree
  with GA-AEM (when installed).

The system is the tTEM GEX of the DAUGAARD case, embedded below so the tests
need no download.

Run only these tests with::

    pytest tests/test_anemone_core.py -v
"""
import importlib.metadata as md
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
anemone = pytest.importorskip("anemone")

import integrate as ig  # noqa: E402
import integrate.anemone_forward as af  # noqa: E402

MIN_VERSION = (0, 1, 1)
N_GATES = 40  # 14 LM + 26 HM used gates of the tTEM GEX below

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="torch has no CUDA device")


def _gaaem_available():
    try:
        import gatdaem1d  # noqa: F401
    except Exception:
        return False
    return True


needs_gaaem = pytest.mark.skipif(not _gaaem_available(),
                                 reason="GA-AEM (gatdaem1d) is not installed")


# ---------------------------------------------------------------------------
# tTEM system: TX07_20231016_2x4_RC20-33.gex (DAUGAARD case)
# ---------------------------------------------------------------------------

def _waveform(name, rows):
    return "\n".join("Waveform%sPoint%02d= %s" % (name, i + 1, r)
                     for i, r in enumerate(rows))


_WF_LM = """-6.7400e-04 -0.000|-6.6372e-04 -0.165|-6.4879e-04 -0.351|-6.3245e-04 -0.508
-6.1499e-04 -0.636|-5.8992e-04 -0.765|-5.6259e-04 -0.862|-5.4090e-04 -0.917
-5.1442e-04 -0.962|-4.9358e-04 -0.987|-4.7400e-04 -1.000|-4.7392e-04 -0.974
-4.7382e-04 -0.863|-4.7374e-04 -0.706|-4.7365e-04 -0.548|-4.7353e-04 -0.375
-4.7337e-04 -0.229|-4.7318e-04 -0.137|-4.7288e-04 -0.075|-4.7246e-04 -0.034
-4.7201e-04 -0.008|-4.7150e-04 -0.000|-2.0000e-04 0.000|-1.8972e-04 0.165
-1.7479e-04 0.351|-1.5845e-04 0.508|-1.4099e-04 0.636|-1.1592e-04 0.765
-8.8592e-05 0.862|-6.6901e-05 0.917|-4.0423e-05 0.962|-1.9577e-05 0.987
0.0000e+00 1.000|7.6606e-08 0.974|1.7545e-07 0.863|2.5947e-07 0.706
3.4843e-07 0.548|4.7199e-07 0.375|6.2521e-07 0.229|8.2290e-07 0.137
1.1194e-06 0.075|1.5395e-06 0.034|1.9893e-06 0.008|2.5000e-06 0.000"""

_WF_HM = """-2.0370e-03 -0.000|-2.0314e-03 -0.087|-2.0102e-03 -0.308|-1.9882e-03 -0.481
-1.9732e-03 -0.574|-1.9525e-03 -0.677|-1.9267e-03 -0.772|-1.8802e-03 -0.880
-1.8412e-03 -0.930|-1.7952e-03 -0.965|-1.7159e-03 -0.993|-1.5870e-03 -1.000
-1.5869e-03 -0.967|-1.5866e-03 -0.813|-1.5864e-03 -0.654|-1.5861e-03 -0.437
-1.5858e-03 -0.211|-1.5855e-03 -0.093|-1.5853e-03 -0.046|-1.5848e-03 -0.019
-1.5825e-03 -0.000|-4.5000e-04 0.000|-4.4437e-04 0.087|-4.2324e-04 0.308
-4.0117e-04 0.481|-3.8615e-04 0.574|-3.6549e-04 0.677|-3.3967e-04 0.772
-2.9319e-04 0.880|-2.5423e-04 0.930|-2.0822e-04 0.965|-1.2887e-04 0.993
0.0000e+00 1.000|1.2788e-07 0.967|4.1701e-07 0.813|6.2829e-07 0.654
8.7294e-07 0.437|1.1732e-06 0.211|1.4734e-06 0.093|1.7403e-06 0.046
2.1629e-06 0.019|4.5000e-06 0.000"""

_GATES = """2.190E-06 3.800E-07 4.000E-06|5.190E-06 4.380E-06 6.000E-06
7.190E-06 6.380E-06 8.000E-06|9.190E-06 8.380E-06 1.000E-05
1.119E-05 1.038E-05 1.200E-05|1.319E-05 1.238E-05 1.400E-05
1.519E-05 1.438E-05 1.600E-05|1.719E-05 1.638E-05 1.800E-05
1.919E-05 1.838E-05 2.000E-05|2.119E-05 2.038E-05 2.200E-05
2.369E-05 2.238E-05 2.500E-05|2.669E-05 2.538E-05 2.800E-05
3.019E-05 2.838E-05 3.200E-05|3.469E-05 3.238E-05 3.700E-05
4.069E-05 3.738E-05 4.400E-05|4.819E-05 4.438E-05 5.200E-05
5.719E-05 5.238E-05 6.200E-05|6.869E-05 6.238E-05 7.500E-05
8.319E-05 7.538E-05 9.100E-05|1.017E-04 9.138E-05 1.120E-04
1.252E-04 1.124E-04 1.380E-04|1.542E-04 1.384E-04 1.700E-04
1.912E-04 1.704E-04 2.120E-04|2.377E-04 2.124E-04 2.630E-04
2.957E-04 2.634E-04 3.280E-04|3.692E-04 3.284E-04 4.100E-04
4.617E-04 4.104E-04 5.130E-04|5.782E-04 5.134E-04 6.430E-04
7.247E-04 6.434E-04 8.060E-04|9.092E-04 8.064E-04 1.012E-03"""


def _rows(s):
    return [r.strip() for r in s.replace("\n", "|").split("|")]


def _channel(n, shift, factor, i0, ngates, repfreq, frontgate, moment, current):
    return f"""[Channel{n}]
RxCoilNumber=1
GateTimeShift={shift}
GateFactor={factor}
RemoveInitialGates={i0}
PrimaryFieldDampingFactor=1
UniformDataSTD=0.03
MeaTimeDelay=0.000E+00
NoGates={ngates}
RepFreq={repfreq}
FrontGateTime={frontgate}
TiBLowPassFilter=1 6.79E+5
TransmitterMoment={moment}
TxApproximateCurrent={current}
SystemResponseConvolution=0
ReceiverPolarizationXYZ=Z
"""


TTEM_GEX = "\n".join([
    "/tTEM gex-file for Tx07 and new RC20 (TX07_20231016_2x4_RC20-33.gex)",
    "[General]",
    "Description=tTEM42",
    "GPSPosition1=            5.85     0.00    -1.20",
    "RxCoilPosition1=        -9.15     0.00    -0.33",
    "TxCoilPosition1=         0.00     0.00    -0.36",
    "AltimeterPosition1=      0.00     0.00     0.00",
    "InclinometerPosition1=   0.00     0.00     0.00",
    "DataType= GroundTEM",
    "LoopType=73",
    "NumberOfTurnsLM=1",
    "NumberOfTurnsHM=1",
    "TxLoopArea=8.00",
    "TxLoopPoint1=   -02.00   -01.00",
    "TxLoopPoint2=    02.00   -01.00",
    "TxLoopPoint3=    02.00    01.00",
    "TxLoopPoint4=   -02.00    01.00",
    "FrontGateDelay=1.9E-06",
    "TXRampLM=",
    _waveform("LM", _rows(_WF_LM)),
    "TXRampHM=",
    _waveform("HM", _rows(_WF_HM)),
    "RxCoilLPFilter1= 0.87 420E+3",
    "RxCoilLPFilter2= 0.87 420E+3",
    "GateNoForPowerLineMonitor=18",
    "FreqForPowerLineMonitor=50.00",
    "CalculateRawDataSTD=1",
    "\n".join("GateTime%02d=%s" % (i + 1, r) for i, r in enumerate(_rows(_GATES))),
    "",
    _channel(1, "-0.90e-6", "0.860", 1, 15, "1055.00", "2.000E-06", "LM", "3.0"),
    _channel(2, "-1.15e-6", "0.970", 4, 30, "315.00", "5.000E-06", "HM", "29.97"),
])


@pytest.fixture(scope="module")
def file_gex(tmp_path_factory):
    f = tmp_path_factory.mktemp("anemone") / "TX07_20231016_2x4_RC20-33.gex"
    f.write_text(TTEM_GEX)
    return str(f)


@pytest.fixture(scope="module")
def models():
    """Smooth random 8-layer resistivity models (ohm-m) and layer thicknesses."""
    rng = np.random.default_rng(42)
    log_rho = 2.0 + np.cumsum(rng.normal(0.0, 0.3, size=(20, 8)), axis=1)
    return 10 ** np.clip(log_rho, 0.0, 3.5), np.full(7, 6.0)


@pytest.fixture(autouse=True)
def _fresh_cache():
    """Each test builds its own anemone setups."""
    af._clear_forward_cache()
    yield
    af._clear_forward_cache()


def _fwd(models, file_gex, **kw):
    M, th = models
    return ig.forward_anemone(M=M, thickness=th, file_gex=file_gex,
                              showInfo=-1, **kw)


# ---------------------------------------------------------------------------
# installed package
# ---------------------------------------------------------------------------

def test_anemone_core_version():
    v = md.version("anemone-core")
    parts = tuple(int(p) for p in v.split(".")[:3])
    assert parts >= MIN_VERSION, "anemone-core %s < %s" % (
        v, ".".join(map(str, MIN_VERSION)))


def test_import_resolves_to_anemone_core():
    """``import anemone`` must load the anemone-core distribution, not e.g.
    an old checkout on PYTHONPATH."""
    dist = md.distribution("anemone-core")
    init = [f for f in dist.files if str(f).replace("\\", "/") == "anemone/__init__.py"]
    assert init, "anemone-core does not ship anemone/__init__.py"
    expected = os.path.realpath(dist.locate_file(init[0]))
    assert os.path.realpath(anemone.__file__) == expected, (
        "import anemone loads %s, not the installed anemone-core (%s)"
        % (anemone.__file__, expected))


def test_anemone_api_used_by_integrate():
    """Names integrate.anemone_forward imports from anemone."""
    from anemone import cdtype, constants  # noqa: F401
    from anemone.forwards import Forward
    from anemone.physics import rte_raiche1999  # noqa: F401
    from anemone.system import (ButterworthFilter, FilterChain, Loop,  # noqa: F401
                                Receiver, Waveform)
    for name in ("setup", "set_device"):
        assert hasattr(Forward, name)
    assert hasattr(FilterChain, "stepoff_response")
    assert hasattr(ButterworthFilter, "poles")


# ---------------------------------------------------------------------------
# anemone-core 0.1.0 bug: repeated cutoffs in FilterChain.stepoff_response
# ---------------------------------------------------------------------------

def _t():
    return torch.cat([torch.tensor([-1e-6, 0.0]),
                      torch.logspace(-9, -4, 200, dtype=torch.float64)])


def test_stepoff_repeated_cutoff():
    """Two identical first-order filters (rx_coil_filter='two_pole').

    0.1.0 raised "All cutoff frequencies in 'omegas' must be distinct".
    Exact step-off of 1/(1+s/w)^2: (1 + w t) exp(-w t) for t > 0, 1 before.
    """
    from anemone.system import ButterworthFilter, FilterChain
    fc = 420e3 / 0.87
    t = _t()
    amp = FilterChain([ButterworthFilter(fc, 1), ButterworthFilter(fc, 1)]
                      ).stepoff_response(t).numpy()
    tn = t.numpy()
    wt = 2 * np.pi * fc * np.clip(tn, 0, None)
    ref = np.where(tn > 0, (1 + wt) * np.exp(-wt), 1.0)
    np.testing.assert_allclose(amp, ref, atol=1e-9)


def _scipy_stepoff(den):
    """Reference step-off of 1/den(s) on an equally spaced grid (scipy needs one)."""
    from scipy.signal import lti
    t = np.linspace(0.0, 2e-5, 4001)
    _, step_on = lti([1.0], den).step(T=t)
    t = t[1:]  # anemone's step-off is 1 at t = 0
    return torch.as_tensor(t), 1.0 - step_on[1:]


def test_stepoff_tib_plus_two_pole():
    """The full default tTEM chain: TiB filter + two equal Rx-coil poles."""
    from anemone.system import ButterworthFilter, FilterChain
    f1, f2 = 6.79e5, 420e3 / 0.87
    w1, w2 = 2 * np.pi * f1, 2 * np.pi * f2
    chain = FilterChain([ButterworthFilter(f1, 1), ButterworthFilter(f2, 1),
                         ButterworthFilter(f2, 1)])
    t, ref = _scipy_stepoff(np.polymul([1 / w1, 1], np.polymul([1 / w2, 1], [1 / w2, 1])))
    np.testing.assert_allclose(chain.stepoff_response(t).numpy(), ref, atol=1e-6)
    np.testing.assert_array_equal(
        chain.stepoff_response(torch.tensor([-1e-6, 0.0])).numpy(), 1.0)


def test_stepoff_damped_lowpass():
    """integrate's _DampedLowPass (rx_coil_filter='damped2') in a FilterChain,
    alone (a GEX without TiB filter) and twice."""
    from anemone.system import FilterChain
    fcut, zeta = 420e3, 0.87
    w = 2 * np.pi * fcut
    den = [1 / w ** 2, 2 * zeta / w, 1]
    flt = af._DampedLowPass(fcut, zeta)
    for chain, d in ((FilterChain([flt]), den),
                     (FilterChain([flt, flt]), np.polymul(den, den))):
        t, ref = _scipy_stepoff(d)
        np.testing.assert_allclose(chain.stepoff_response(t).numpy(), ref, atol=1e-6)


# ---------------------------------------------------------------------------
# forward_anemone
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gate_integration", ["centre", "boxcar"])
@pytest.mark.parametrize("rx_coil_filter", ["two_pole", "damped2"])
def test_forward_runs(models, file_gex, rx_coil_filter, gate_integration):
    D = _fwd(models, file_gex, rx_coil_filter=rx_coil_filter,
             gate_integration=gate_integration)
    assert D.shape == (models[0].shape[0], N_GATES)
    assert np.all(np.isfinite(D))
    # early LM and HM gates: positive, decaying dB/dt
    for sl in (slice(1, 10), slice(14, 24)):
        assert np.all(D[:, sl] > 0)
        assert np.all(np.diff(D[:, sl], axis=1) < 0)


@pytest.mark.parametrize("gate_integration", [
    "centre",
    "boxcar",  # broken in anemone-core 0.1.1 from PyPI, fixed by anemone#2
])
def test_dual_moment_matches_single_moments(file_gex, gate_integration):
    """The combined LM+HM Forward (Forward.__add__) must give the same data
    as each moment's own Forward."""
    system = af.gex_to_anemone_system(file_gex, gate_integration=gate_integration)
    fwr, slices = af._build_forward(system, None, "cpu")
    rho = torch.tensor([[100.0], [20.0], [300.0]], dtype=torch.float64)
    thick = torch.tensor([[10.0], [20.0]], dtype=torch.float64)
    with torch.no_grad():
        d = fwr(rho, thick).numpy().ravel()
        for s, part in zip(slices, fwr.moment_parts):
            np.testing.assert_allclose(d[s], part(rho, thick).numpy().ravel(),
                                       rtol=1e-3)


def test_forward_single_model(models, file_gex):
    M, th = models
    D1 = ig.forward_anemone(M=M[0], thickness=th, file_gex=file_gex, showInfo=-1)
    D = _fwd(models, file_gex)
    np.testing.assert_allclose(np.squeeze(D1), D[0], rtol=1e-10)


def test_forward_halfspace_scaling(file_gex):
    """Higher resistivity -> faster decay: late gates drop."""
    th = np.full(3, 10.0)
    M = np.array([[10.0] * 4, [100.0] * 4, [1000.0] * 4])
    D = ig.forward_anemone(M=M, thickness=th, file_gex=file_gex, showInfo=-1)
    late = D[:, 30]
    assert late[0] > late[1] > late[2] > 0


@needs_gaaem
def test_forward_matches_gaaem(models, file_gex, tmp_path, monkeypatch):
    """anemone and GA-AEM agree to a few percent on gates 2..40.

    The first LM gate is left out: anemone-core includes the filtered
    primary field there (GA-AEM does not).
    """
    monkeypatch.chdir(tmp_path)  # GA-AEM writes .stm files to the cwd
    M, th = models
    D_an = _fwd(models, file_gex)
    D_ga = ig.forward_gaaem(C=1.0 / M, thickness=th, file_gex=file_gex,
                            showInfo=-1)
    assert D_ga.shape == D_an.shape
    r = np.abs(D_an[:, 1:] / D_ga[:, 1:] - 1)
    assert np.median(r) < 0.01, np.median(r)
    assert np.percentile(r, 95) < 0.05, np.percentile(r, 95)


def test_varying_height_rescale(models, file_gex):
    """Varying Tx height: one exact setup per height vs one rescaled setup.

    Also runs the dh=0 guard in _forward_rescaled_height, which fails if the
    rescaled evaluation no longer reproduces anemone's Forward.
    """
    M, th = models
    h = np.repeat([0.4, 1.0, 1.7, 2.5], M.shape[0] // 4)
    D_exact = ig.forward_anemone(M=M, thickness=th, file_gex=file_gex,
                                 tx_height=h, altitude_bin_width=0, showInfo=-1)
    af._clear_forward_cache()
    D_res = ig.forward_anemone(M=M, thickness=th, file_gex=file_gex,
                               tx_height=h, rescale_height=True, showInfo=-1)
    assert np.all(np.isfinite(D_res))
    err = np.abs(D_res - D_exact) / np.abs(D_exact).max(axis=1, keepdims=True)
    print("rescale_height: max error %.2e of peak per height:" % err.max(),
          [float("%.1e" % err[h == hh].max()) for hh in np.unique(h)])
    assert err.max() < 1e-3, err.max()


# ---------------------------------------------------------------------------
# anemone-core 0.1.0 bug: CUDA device handling
# ---------------------------------------------------------------------------

@needs_cuda
@pytest.mark.parametrize("rx_coil_filter", ["two_pole", "damped2"])
def test_forward_cuda_matches_cpu(models, file_gex, rx_coil_filter):
    """0.1.0 failed on CUDA with "Expected all tensors to be on the same device"."""
    D_cpu = _fwd(models, file_gex, rx_coil_filter=rx_coil_filter, device="cpu")
    D_gpu = _fwd(models, file_gex, rx_coil_filter=rx_coil_filter, device="cuda")
    scale = np.abs(D_cpu).max(axis=1, keepdims=True)
    np.testing.assert_allclose(D_gpu / scale, D_cpu / scale, atol=1e-8)


@needs_cuda
def test_forward_set_device(file_gex):
    """A Forward built on the CPU and moved with set_device('cuda')."""
    system = af.gex_to_anemone_system(file_gex)
    fwr, _ = af._build_forward(system, None, "cpu")
    # (layers, models), as integrate.anemone_forward._run_batched calls it
    rho = torch.tensor([[100.0, 10.0, 300.0], [30.0, 30.0, 30.0]],
                       dtype=torch.float64).T
    thick = torch.tensor([[10.0, 20.0], [10.0, 20.0]], dtype=torch.float64).T
    with torch.no_grad():
        d_cpu = fwr(rho, thick).cpu().numpy()
        fwr.set_device("cuda")
        try:
            d_gpu = fwr(rho.cuda(), thick.cuda()).cpu().numpy()
        finally:
            fwr.set_device("cpu")
    np.testing.assert_allclose(d_gpu, d_cpu, rtol=1e-8,
                               atol=1e-8 * np.abs(d_cpu).max())


# ---------------------------------------------------------------------------
# prior_data_anemone
# ---------------------------------------------------------------------------

def test_prior_data_anemone(file_gex, tmp_path, monkeypatch):
    import h5py
    monkeypatch.chdir(tmp_path)
    N = 30
    f_prior_h5 = ig.prior_model_layered(N=N, NLAY_min=3, NLAY_max=5, z_max=60,
                                        RHO_min=5, RHO_max=500, showInfo=-1,
                                        f_prior_h5=str(tmp_path / "PRIOR.h5"))
    f_pd = ig.prior_data_anemone(f_prior_h5, file_gex, N=N, showInfo=-1)
    with h5py.File(f_pd, "r") as f:
        D = f["D1"][:]
        nm = f["M1"].shape[0]
    assert D.shape == (nm, N_GATES)
    assert np.all(np.isfinite(D))
