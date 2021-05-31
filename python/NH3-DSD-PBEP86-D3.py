import ctypes
import numpy as np
from pyscf import lib
from pyscf import gto, scf, dft, mp

libdftd3 = np.ctypeslib.load_library('libdftd3.so', "../lib")


def get_energy_bDH(mol, xc_scf, c_pt2, c_os, c_ss):
    mf = dft.KS(mol, xc=xc_scf)
    mf.conv_tol_grad = 1e-10
    mf.grids.atom_grid = (99, 590)
    mf._numint.libxc = dft.xcfun   # I'm not sure whether libxc have discrepency for P86 corr
    mf.run()
    if not mf.converged:
        raise ValueError("SCF not converged.")
    eng_scf = mf.e_tot
    # handle SS/OS partition
    mf_pt2 = mp.MP2(mf).run()
    t2 = mf_pt2.t2
    nocc = t2.shape[0]
    e = mf.mo_energy
    d2 = (+ e[:nocc, None, None, None] + e[None, :nocc, None, None]
          - e[None, None, nocc:, None] - e[None, None, None, nocc:])
    eng_os = np.einsum("ijab, ijab, ijab ->", t2, t2, d2)
    eng_ss = eng_os - np.einsum("ijab, ijba, ijab ->", t2, t2, d2)
    eng_pt2 = c_pt2 * (c_os * eng_os + c_ss * eng_ss)
    #      Total energy       SCF part PT2 part  (without D3(BJ))
    return eng_scf + eng_pt2, eng_scf, eng_pt2


if __name__ == "__main__":

    mol = gto.Mole(atom="N 0. 0. 0.; H .9 0. 0.; H 0. 1. 0.; H 0. 0. 1.1", basis="cc-pVDZ", verbose=0).build()

    # D3(BJ) calculation
    params = np.array([0.48, 0, 0, 5.6, 0])
    version = 4
    coords = np.asarray(mol.atom_coords(), order="F")
    itype = np.asarray(mol.atom_charges(), order="F")
    edisp = np.zeros(1)
    grad = np.zeros((mol.natm, 3)) 
    libdftd3.wrapper_params(
        ctypes.c_int(mol.natm),                  # natoms
        coords.ctypes.data_as(ctypes.c_void_p),  # coords
        itype.ctypes.data_as(ctypes.c_void_p),   # itype
        params.ctypes.data_as(ctypes.c_void_p),  # params
        ctypes.c_int(version),                   # version
        edisp.ctypes.data_as(ctypes.c_void_p),   # edisp
        grad.ctypes.data_as(ctypes.c_void_p))    # grads
    # Python   -0.0004626305459352599
    # Gaussian -0.0004626305
    print("Energy of D3(BJ): ", float(edisp))
    
    # DSD-PBEP86 calculation
    eng_tot, eng_scf, eng_pt2 = get_energy_bDH(mol, "0.69*HF + 0.31*PBE, 0.44*P86", 1, 0.52, 0.22)
    # Python   -56.300186987892154
    # Gaussian -56.3001820336
    print("SCF total energy: ", float(eng_scf + edisp))
    # Python   -56.39730652290786
    # Gaussian -56.397301565299
    print("bDH total energy: ", float(eng_tot + edisp))

"""
Gaussian Input Card

#p DSDPBEP86(Full)/cc-pVDZ Int(Grid=99590) NoSymm

NH3 DSD-PBEP86-D3(BJ) Single Point Energy

0 1
N  0.  0.  0.
H  0.9 0.  0.
H  0.  1.  0.
H  0.  0.  1.1

! Dispersion Correction: -0.0004626305    a.u.
! SCF Done Energy      : -56.3001820336   a.u.
! PT2 Corrected Energy : -56.397301565299 a.u.
"""
