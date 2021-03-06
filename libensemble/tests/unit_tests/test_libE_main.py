import sys, time, os
import numpy as np
import pytest
import mock

#sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from libensemble.libE import check_inputs, libE
import libensemble.tests.unit_tests.setup as setup
from mpi4py import MPI

al = {}
libE_specs = {'comm':MPI.COMM_WORLD}
fname_abort = 'libE_history_at_abort_0.npy'


def test_manager_exception():
    try:
        os.remove(fname_abort)
    except OSError as e:
        pass
    with mock.patch('libensemble.libE.manager_main') as managerMock:
        managerMock.side_effect = Exception
        with mock.patch('libensemble.libE.comms_abort') as abortMock:
            abortMock.side_effect = Exception
            with pytest.raises(Exception, message='Expected exception'):
                libE({'out':[('f',float)]},{'out':[('x',float)]},{'sim_max':1},libE_specs={'comm': MPI.COMM_WORLD})
            # Check npy file dumped
            assert os.path.isfile(fname_abort), "History file not dumped"
            os.remove(fname_abort)


def test_exception_raising_manager():
    # Intentionally running without sim_specs['in'] to test exception raising (Fails)
    with mock.patch('libensemble.libE.comms_abort') as abortMock:
        abortMock.side_effect = Exception
        with pytest.raises(Exception, message='Expected exception'):
            H,_,_ = libE({'out':[('f',float)]},{'out':[('x',float)]},{'sim_max':1},libE_specs={'comm': MPI.COMM_WORLD})


# def test_exception_raising_worker():
#     # Intentionally running without sim_specs['in'] to test exception raising (Fails)
#     H,_,_ = libE({'out':[('f',float)]},{'out':[('x',float)]},{'sim_max':1},libE_specs={'comm': MPI.COMM_WORLD})
#     assert H==[]

def test_checking_inputs():

    # Don't take more points than there is space in history.
    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()

    H0 = np.zeros(3,dtype=sim_specs['out'] + gen_specs['out'] + [('returned',bool)])
    # Should fail because H0 has points with 'return'==False
    try:
        check_inputs(libE_specs,al, sim_specs, gen_specs, exit_criteria,H0)
    except AssertionError:
        assert 1
    else:
        assert 0

    # Should not fail
    H0['returned']=True
    check_inputs(libE_specs,al, sim_specs, gen_specs, exit_criteria,H0)

    # Removing 'returned' and then testing again.
    H0 = rmfield( H0, 'returned')
    check_inputs(libE_specs,al, sim_specs, gen_specs, exit_criteria,H0)

    # Should fail because H0 has fields not in H
    H0 = np.zeros(3,dtype=sim_specs['out'] + gen_specs['out'] + [('bad_name',bool),('bad_name2',bool)])
    try:
        check_inputs(libE_specs,al, sim_specs, gen_specs, exit_criteria,H0)
    except AssertionError:
        assert 1
    else:
        assert 0

def rmfield( a, *fieldnames_to_remove ):
        return a[ [ name for name in a.dtype.names if name not in fieldnames_to_remove ] ]

if __name__ == "__main__":
    test_manager_exception()
    test_exception_raising_manager()
    test_checking_inputs()
