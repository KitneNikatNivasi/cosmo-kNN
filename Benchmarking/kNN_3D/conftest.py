import pytest

import numpy as np

def pytest_addoption(parser):
    parser.addoption("--NSIDE", action="store", default="256")
    parser.addoption("--n_tracer", action="store", default="20000")
    parser.addoption("--k_List", action="store", default="1")
    parser.addoption("--rounds", action="store", default="5")
    parser.addoption("--warmup", action="store", default="1")
    parser.addoption("--NGRID", action="store", default="256")
    parser.addoption("--BoxSize", action="store", default="1000.0")
    parser.addoption("--R_list", action="store", default="10 20 40")
    parser.addoption("--threads", action="store", default="1")
    parser.addoption("--n_query", action="store", default="5000")
    parser.addoption("--kA_kB_List", action="store", default="1,1")
    parser.addoption("--n_reals", action="store", default="3")
    parser.addoption("--FieldConstPercThreshold", action="store", default="75.0")

@pytest.fixture(scope="session")
def NSIDE(pytestconfig):
    return int(pytestconfig.getoption("NSIDE"))

@pytest.fixture(scope="session")
def n_tracer(pytestconfig):
    return int(pytestconfig.getoption("n_tracer"))

@pytest.fixture(scope="session")
def k_List(pytestconfig):
    return list(np.array(pytestconfig.getoption("k_List").split()).astype(int))

@pytest.fixture(scope="session")
def rounds(pytestconfig):
    return int(pytestconfig.getoption("rounds"))

@pytest.fixture(scope="session")
def warmup(pytestconfig):
    return int(pytestconfig.getoption("warmup"))

@pytest.fixture(scope="session")
def NGRID(pytestconfig):
    return int(pytestconfig.getoption("NGRID"))

@pytest.fixture(scope="session")
def BoxSize(pytestconfig):
    return float(pytestconfig.getoption("BoxSize"))

@pytest.fixture(scope="session")
def R_list(pytestconfig):
    return list(np.array(pytestconfig.getoption("R_list").split()).astype(float))

@pytest.fixture(scope="session")
def threads(pytestconfig):
    return int(pytestconfig.getoption("threads"))

@pytest.fixture(scope="session")
def n_query(pytestconfig):
    return int(pytestconfig.getoption("n_query"))

@pytest.fixture(scope="session")
def kA_kB_List(pytestconfig):
    # space-separated pairs, each pair comma-separated, e.g. "1,1 1,2 2,1"
    pairs = pytestconfig.getoption("kA_kB_List").split()
    return [tuple(int(v) for v in pair.split(",")) for pair in pairs]

@pytest.fixture(scope="session")
def n_reals(pytestconfig):
    return int(pytestconfig.getoption("n_reals"))

@pytest.fixture(scope="session")
def FieldConstPercThreshold(pytestconfig):
    return float(pytestconfig.getoption("FieldConstPercThreshold"))
