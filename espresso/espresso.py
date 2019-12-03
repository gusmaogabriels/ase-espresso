# -*- coding: utf-8 -*-

# ****************************************************************************
# Original work Copyright (C) 2013-2015 SUNCAT
# Modified work Copyright 2015-2017 Lukasz Mentel
#
# This file is distributed under the terms of the
# GNU General Public License. See the file 'COPYING'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
# ****************************************************************************

from __future__ import print_function, absolute_import, unicode_literals

from builtins import (super, range, zip, round, int, object)

from builtins import str as newstr

import os
import re
import tarfile
import atexit
import shutil
import subprocess
import numpy as np
from collections import OrderedDict
from io import open
try:
    from path import Path
except:
    from pypath import Path
import logging

import pexpect

from ase.calculators.calculator import FileIOCalculator
from ase.units import Hartree, Rydberg, Bohr

from .utils import speciestuple, num2str, bool2str, convert_constraints
from .siteconfig import SiteConfig, preserve_cwd

__version__ = '0.3.3'

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

all_changes = ['positions', 'numbers', 'cell', 'pbc',
               'initial_charges', 'initial_magmoms']


class SCFConvergenceError(Exception):
    pass


class SCFMaxIterationsError(Exception):
    pass


class Espresso(FileIOCalculator, object):
    """
    ASE interface for Quantum Espresso

    Args:
        atoms (``ase.Atoms``) :
            list of atoms object to be attached to calculator
            atoms.set_calculator can be used instead

        beefensemble (bool) :
            calculate basis energies for ensemble error estimates based on
            the BEEF-vdW functional, defaults to `False`

        calcstress (bool) :
            If True, calculate stress, defaults to ``False``

        calculation (str) :
            Specifies the type of calcualtion, can be one of `relax`, `scf`,
            `nscf` corresponding Quantum Espresso standard modes

        cell_dofree (None) :
            partially fix lattice vectors

        cell_dynamics (str) :
            algorithm (e.g. `BFGS`) to be used for Espresso-internal unit-cell
            optimization

        cell_factor (float) :
            should be >>1 if unit-cell volume is expected to shrink a lot
            during relaxation (would be more efficient to start with a better
            guess)

        charge (None) :
            overrides tot_charge (ase 3.7+ compatibility)

        constr_tol (float) :
            constraint tolerance for Espresso-internal relaxation

        convergence (``dict``) :
            Electronic convergence criteria and diag. and mixing algorithms.

            Additionally, a preconditioner for the mixing algoritms can be
            specified, e.g. ``'mixing_mode': 'local-TF'`` or
            ``'mixing_mode': 'TF'``.

            defaults to ``{'energy': 1e-6, 'mixing':0.7, 'maxsteps':100, 'diag':'david'}``

        dipole (`dict`) :
            If `status` is ``True``, turn on dipole correction; then by
            default, the dipole correction is applied along the z-direction,
            and the dipole is put in the center of the vacuum region (taking
            periodic boundary conditions into account).

            defaults to ``{'status': False}``

            This can be overridden with

            - `edir` equal to 1, 2, or 3 for x-, y-, or z-direction
            - `emaxpos`, float percentage wrt. unit cell where dip. correction
              potential will be max.
            - `eopreg`, float percentage wrt. unit cell where potential decreases
            - `eamp` equal to 0 (by default) if non-zero overcompensate dipole: i.e. apply
              a field

        dpress (float) :
            convergence limit towards target pressure

        dw (float) :
            charge-density cut-off in eV, defaults to 10.0 * `pw`

        exx_fraction (float) :
            Default depends on hybrid functional chosen.

        exxdiv_treatment (str) :
            Method to treat Coulomb potential divergence for small q,
            defaults to `gygi-baldereschi`

        ecutvcut (float) :
            Cut-off for ``exxdiv_treatment``

        fix_magmom (bool) :
            If `True`, fix total magnetization to current value,
            defaults to `False`

        fft_grid (tuple) :
            a tuple of fft grid points ``(nr1, nr2, nr3)`` for q.e.
            useful for series of calculations with changing cell size
            (e.g. lattice constant optimization) uses Quantum Esoresso default
            if not specified

            If specified, sets the keywrds `nr1`, `nr2`, `nr3` in Quantum Espresso input

        fmax (float) :
            max force limit for Espresso-internal relaxation (eV/Angstrom),
            defaults to 0.05

        fw (float) :
            plane-wave cutoff for evaluation of EXX in eV, defaults to ``None``

        ion_dynamics (str) :
            Type of relaxation, detaults to `ase3`.

            - `ase3` - only possible with dynamic communication between
              Quantum Espresso and python in this case ASE updates coordinates
              during relaxation
            - `relax` - and other Quantum Espresso standard relaxation modes;
              Quantum Espresso own algorithms for structural optimization are
              used

            Obtaining Quantum Espresso with the ase3 relaxation extensions is
            highly recommended, since it allows for using ase's optimizers
            without loosing efficiency

            .. code-block:: bash

               svn co --username anonymous http://qeforge.qe-forge.org/svn/q-e/branches/espresso-dynpy-beef

        isolated (str) :
            invoke an `assume_isolated` method for screening long-range
            interactions across 3D supercells, particularly electrostatics.
            Very useful for charged molecules and charged surfaces,
            but also improves convergence wrt. vacuum space for neutral
            molecules.

            - `makov-payne`, `mp` - only cubic systems.
            - `dcc` - don't use.
            - `martyna-tuckerman`, `mt` - method of choice for molecules,
              works for any supercell geometry.
            - `esm` - Effective Screening Medium Method for surfaces and
              interfaces.

        kpts (`tuple` or str) :
            k-point grid sub-divisions, k-point grid density,
            explicit list of k-points, or simply `gamma` for gamma-point only,
            defaults to ``(1, 1, 1)``

        kptshift (tuple) :
            shift of k-point grid, defaults to ``(0, 0, 0)``

        nbands (int) :
            number of bands, if negative: -n extra bands, defaults to -10

        noinv (False) :

        nosym (False) :

        nosym_evc (False) :

        no_t_rev (False) :
            turn off corresp. symmetries

        noncollinear (bool):
            Non-collinear magnetism, defaults to `False`

        nqx1 (int) :
            3D mesh for q=k1-k2 sampling of Fock operator. Can be smaller
            than number of k-points. Defaults to ``None``

        nqx2 (int) :
            3D mesh for q=k1-k2 sampling of Fock operator. Can be smaller
            than number of k-points. Defaults to ``None``

        nqx3 (int) :
            3D mesh for q=k1-k2 sampling of Fock operator. Can be smaller
            than number of k-points. Defaults to ``None``

        occupations (str) :
            Controls how Kohn-Sham states are occupied.
            Possible values: `smearing`, `fixed` (molecule or insulator),
            or `tetrahedra`.

        outdir (str) :
            directory where Espresso's output is collected,
            defaults to `qe_$JOBID`

        output (`dict`) :
            control how much io is used by espresso

            defaults to ``{'disk_io': 'default', 'avoidio': False,
            'removewf': True, 'removesave': False, 'wf_collect': False}``

            - `disk_io` - how often espresso writes wavefunctions to disk
            - `avoidio` - will overwrite disk_io parameter if ``True``
            - `removewf`- if ``True`` wave functions are deleted in scratch
              area before job is done and data is copied back to submission
              directory
            - `removesave` if ``True`` the whole ``.save`` directory is
              deleted in scratch area

        parflags (None)
            Parallelization flags for Quantum Espresso.
            E.g. parflags='-npool 2' will distribute k-points (and spin if
            spin-polarized) over two nodes.

        press (float) :
            target pressure

        printensemble (bool) :
            let Espresso itself calculate 2000 ensemble energies,
            defaults to `False`

        psppath (str) :
            Directory containing the pseudo-potentials or paw-setups to be used
            The ase-espresso interface expects all pot. files to be of the type
            element.UPF (e.g. H.UPF).
            If ``None``, the directory pointed to be ESP_PSP_PATH is used.

        pw (float) :
            plane-wave cut-off in eV, defaults to 350.0

        screening_parameter (float) :
            Screening parameter for HSE-like functional, defaults to 0.106

        sigma (float) :
            smearing width in eV, defaults to 0.1

        site (`siteconfig.SiteConfig`) :
            Site configuration deifinig how to execute pw.x in batch
            environment

        smearing (str) :
            method for Fermi surface smearing

            - `fd`, `Fermi-Dirac` for Fermi-Dirac
            - `mv`, `Marzari-Vanderbilt` for Marzari-Vanderbilt cold smearing
            - `gauss`, `gaussian` for Gaussian smearing
            - `mp`, `Methfessel-Paxton` for Methfessel-Paxton

        spinorbit (bool) :
            If True, spin-orbit coupling is considered.
            Make sure to provide j-dependent pseudo-potentials in psppath
            for those elements where spin-orbit coupling is important,
            defaults to `False`

        spinpol (bool) :
            If `True`, calculation is spin-polarized, defaults to `False`

        startingpot (str) :
            Iinitial guess for the potential.

            defaults to `atomic` - use superposition of atomic orbitals,
            `file` - construct potential from `charge-density.dat`
            Can be used with :py:meth:`load_chg` and :py:meth:`save_chg`
            methods.

        startingwfc (str) :
            Initial guess for the wave function

            defaults to `atomic`, other options: `atomic+random` or `random`,
            `file` - reload wave functions from other calculations.

            See also:
                :py:meth:`load_wf` and :py:meth:`save_wf` methods

        U (`list` or `dict`) :
            specify Hubbard U values (in eV)

            - ``U`` can be specified as a list of values for each atom
            - ``U`` can be specified as a dictionary, e.g. ``U={'Fe':3.5}``

            U values are assigned to angular momentum channels according to
            Espresso's hard-coded defaults, i.e. l=2 for transition metals,
            l=1 for oxygen, etc.

        J (`list` or `dict`) :
            specify exchange J values (in eV), see :py:attr:`U` parameter

        U_alpha (`list` or `dict`) :
            U_alpha (in eV), see :py:attr:`U` parameter

        U_projection_type ('atomic')
            type of projectors for calculating density matrices in DFT+U
            schemes

        tot_charge (int) :
            charge the unit cell, +1 means 1 e missing, -1 means 1 extra e

        tot_magnetization (int or str) :
            Fix total magnetization, -1 means unspecified/free,
            `hund` means Hund's rule for each atom, defaults to -1

        txt (str) :
            If not `None`, direct Espresso's output to a different file than
            `outdir/log`

        verbose (str) :
            verbosity level, can be `high` or `low`

        xc (str) :
            xc-functional to be used, defaults to `PBE`

    """

    implemented_properties = ['energy', 'forces', 'free_energy', 'magmom',
                              'magmoms', 'stress']

    default_parameters = []

    def __init__(self,
                 atoms=None,
                 pw=350.0,
                 dw=None,
                 fw=None,
                 nbands=-10,
                 kpts=(1, 1, 1),
                 kptshift=(0, 0, 0),
                 fft_grid=None,
                 calculation='relax',
                 ion_dynamics='ase3',
                 nstep=None,
                 constr_tol=None,
                 fmax=0.05,
                 cell_dynamics=None,
                 press=None,
                 dpress=None,
                 cell_factor=None,
                 cell_dofree=None,
                 dontcalcforces=False,
                 nosym=False,
                 noinv=False,
                 nosym_evc=False,
                 no_t_rev=False,
                 xc='PBE',
                 beefensemble=False,
                 printensemble=False,
                 psppath=None,
                 spinpol=False,
                 noncollinear=False,
                 spinorbit=False,
                 outdir=None,
                 txt=None,
                 calcstress=False,
                 smearing='fd',
                 sigma=0.1,
                 fix_magmom=False,
                 isolated=None,
                 U=None,
                 J=None,
                 U_alpha=None,
                 U_projection_type='atomic',
                 nqx1=None,
                 nqx2=None,
                 nqx3=None,
                 exx_fraction=None,
                 screening_parameter=None,
                 exxdiv_treatment=None,
                 ecutvcut=None,
                 tot_charge=None,
                 charge=None,
                 tot_magnetization=-1,
                 occupations='smearing',
                 dipole={'status': False},
                 field={'status': False},
                 output={'disk_io': 'default',
                         'avoidio': False,
                         'removewf': True,
                         'removesave': False,
                         'wf_collect': False},
                 convergence={'energy': 1e-6,
                              'mixing': 0.7,
                              'maxsteps': 100,
                              'diag': 'david'},
                 startingpot=None,
                 startingwfc=None,
                 ion_positions=None,
                 parflags=None,
                 alwayscreatenewarrayforforces=True,
                 verbose='low',
                 # automatically generated list of parameters
                 # some coincide with ase-style names
                 iprint=None,
                 tstress=None,
                 tprnfor=None,
                 dt=None,
                 lkpoint_dir=None,
                 max_seconds=None,
                 etot_conv_thr=None,
                 forc_conv_thr=None,
                 tefield=None,
                 dipfield=None,
                 lelfield=None,
                 nberrycyc=None,
                 lorbm=None,
                 lberry=None,
                 gdir=None,
                 nppstr=None,
                 nbnd=None,
                 ecutwfc=None,
                 ecutrho=None,
                 ecutfock=None,
                 force_symmorphic=None,
                 use_all_frac=None,
                 one_atom_occupations=None,
                 starting_spin_angle=None,
                 degauss=None,
                 nspin=None,
                 ecfixed=None,
                 qcutz=None,
                 q2sigma=None,
                 x_gamma_extrapolation=None,
                 lda_plus_u=None,
                 lda_plus_u_kind=None,
                 edir=None,
                 emaxpos=None,
                 eopreg=None,
                 eamp=None,
                 clambda=None,
                 report=None,
                 lspinorb=None,
                 esm_w=None,
                 esm_efield=None,
                 esm_nfit=None,
                 london=None,
                 london_s6=None,
                 london_rcut=None,
                 xdm=None,
                 xdm_a1=None,
                 xdm_a2=None,
                 electron_maxstep=None,
                 scf_must_converge=None,
                 conv_thr=None,
                 adaptive_thr=None,
                 conv_thr_init=None,
                 conv_thr_multi=None,
                 mixing_beta=None,
                 mixing_ndim=None,
                 mixing_fixed_ns=None,
                 ortho_para=None,
                 diago_thr_init=None,
                 diago_cg_maxiter=None,
                 diago_david_ndim=None,
                 diago_full_acc=None,
                 efield=None,
                 tqr=None,
                 remove_rigid_rot=None,
                 tempw=None,
                 tolp=None,
                 delta_t=None,
                 nraise=None,
                 refold_pos=None,
                 upscale=None,
                 bfgs_ndim=None,
                 trust_radius_max=None,
                 trust_radius_min=None,
                 trust_radius_ini=None,
                 w_1=None,
                 w_2=None,
                 wmass=None,
                 press_conv_thr=None,
                 site=None,
                 force_consistent=False
                 ):

        self.pw = pw
        self.dw = dw
        self.fw = fw

        if self.dw is None:
            self.dw = 10.0 * self.pw

        if self.dw < self.pw:
            raise ValueError('<dw> smaller than <pw>: {0:.2f} < {1:.2f}'.format(self.dw, self.pw))

        self.nbands = nbands
        self.kpts = kpts
        self.kptshift = kptshift
        self.fft_grid = fft_grid  # RK
        self.calculation = calculation

        if self.calculation in ['scf', 'nscf', 'bands', 'hund']:
            self.ion_dynamics = None
        else:
            self.ion_dynamics = ion_dynamics

        self.nstep = nstep
        self.constr_tol = constr_tol
        self.fmax = fmax
        self.cell_dynamics = cell_dynamics
        self.press = press
        self.dpress = dpress
        self.cell_factor = cell_factor
        self.cell_dofree = cell_dofree
        self.dontcalcforces = dontcalcforces
        self.nosym = nosym
        self.noinv = noinv
        self.nosym_evc = nosym_evc
        self.no_t_rev = no_t_rev
        self.xc = xc
        self.beefensemble = beefensemble
        self.printensemble = printensemble
        self.smearing = smearing
        self.sigma = sigma
        self.spinpol = spinpol
        self.noncollinear = noncollinear
        self.spinorbit = spinorbit
        self.fix_magmom = fix_magmom
        self.isolated = isolated

        if charge is None:
            self.tot_charge = tot_charge
        else:
            self.tot_charge = charge

        self.tot_magnetization = tot_magnetization
        self.occupations = occupations
        self.outdir = outdir
        self.calcstress = calcstress
        self.psppath = psppath
        self.dipole = dipole
        self.field = field
        self.output = output
        self.convergence = convergence
        self.startingpot = startingpot
        self.startingwfc = startingwfc
        self.ion_positions = ion_positions
        self.verbose = verbose
        self.U = U
        self.J = J
        self.U_alpha = U_alpha
        self.U_projection_type = U_projection_type
        self.nqx1 = nqx1
        self.nqx2 = nqx2
        self.nqx3 = nqx3
        self.exx_fraction = exx_fraction
        self.screening_parameter = screening_parameter
        self.exxdiv_treatment = exxdiv_treatment
        self.ecutvcut = ecutvcut
        self.newforcearray = alwayscreatenewarrayforforces
        if parflags is None:
            self.parflags = ''
        else:
            self.parflags = parflags
        self.txt = txt

        self.mypath = os.path.abspath(os.path.dirname(__file__))

        self.atoms = None
        self.sigma_small = 1e-13
        self.started = False
        self.got_energy = False
        self.only_init = False

        # automatically generated list
        self.iprint = iprint
        self.tstress = tstress
        self.tprnfor = tprnfor
        self.dt = dt
        self.lkpoint_dir = lkpoint_dir
        self.max_seconds = max_seconds
        self.etot_conv_thr = etot_conv_thr
        self.forc_conv_thr = forc_conv_thr
        self.tefield = tefield
        self.dipfield = dipfield
        self.lelfield = lelfield
        self.nberrycyc = nberrycyc
        self.lorbm = lorbm
        self.lberry = lberry
        self.gdir = gdir
        self.nppstr = nppstr
        self.nbnd = nbnd
        self.ecutwfc = ecutwfc
        self.ecutrho = ecutrho
        self.ecutfock = ecutfock
        self.force_symmorphic = force_symmorphic
        self.use_all_frac = use_all_frac
        self.one_atom_occupations = one_atom_occupations
        self.starting_spin_angle = starting_spin_angle
        self.degauss = degauss
        self.nspin = nspin
        self.ecfixed = ecfixed
        self.qcutz = qcutz
        self.q2sigma = q2sigma
        self.x_gamma_extrapolation = x_gamma_extrapolation
        self.lda_plus_u = lda_plus_u
        self.lda_plus_u_kind = lda_plus_u_kind
        self.edir = edir
        self.emaxpos = emaxpos
        self.eopreg = eopreg
        self.eamp = eamp
        self.clambda = clambda
        self.report = report
        self.lspinorb = lspinorb
        self.esm_w = esm_w
        self.esm_efield = esm_efield
        self.esm_nfit = esm_nfit
        self.london = london
        self.london_s6 = london_s6
        self.london_rcut = london_rcut
        self.xdm = xdm
        self.xdm_a1 = xdm_a1
        self.xdm_a2 = xdm_a2
        self.electron_maxstep = electron_maxstep
        self.scf_must_converge = scf_must_converge
        self.conv_thr = conv_thr
        self.adaptive_thr = adaptive_thr
        self.conv_thr_init = conv_thr_init
        self.conv_thr_multi = conv_thr_multi
        self.mixing_beta = mixing_beta
        self.mixing_ndim = mixing_ndim
        self.mixing_fixed_ns = mixing_fixed_ns
        self.ortho_para = ortho_para
        self.diago_thr_init = diago_thr_init
        self.diago_cg_maxiter = diago_cg_maxiter
        self.diago_david_ndim = diago_david_ndim
        self.diago_full_acc = diago_full_acc
        self.efield = efield
        self.tqr = tqr
        self.remove_rigid_rot = remove_rigid_rot
        self.tempw = tempw
        self.tolp = tolp
        self.delta_t = delta_t
        self.nraise = nraise
        self.refold_pos = refold_pos
        self.upscale = upscale
        self.bfgs_ndim = bfgs_ndim
        self.trust_radius_max = trust_radius_max
        self.trust_radius_min = trust_radius_min
        self.trust_radius_ini = trust_radius_ini
        self.w_1 = w_1
        self.w_2 = w_2
        self.wmass = wmass
        self.press_conv_thr = press_conv_thr
        self.force_consistent = force_consistent
        self.defaults = dict(atoms=None,
                 pw=350.0,
                 dw=None,
                 fw=None,
                 nbands=-10,
                 kpts=(1, 1, 1),
                 kptshift=(0, 0, 0),
                 fft_grid=None,
                 calculation='relax',
                 ion_dynamics='ase3',
                 nstep=None,
                 constr_tol=None,
                 fmax=0.05,
                 cell_dynamics=None,
                 press=None,
                 dpress=None,
                 cell_factor=None,
                 cell_dofree=None,
                 dontcalcforces=False,
                 nosym=False,
                 noinv=False,
                 nosym_evc=False,
                 no_t_rev=False,
                 xc='PBE',
                 beefensemble=False,
                 printensemble=False,
                 psppath=None,
                 spinpol=False,
                 noncollinear=False,
                 spinorbit=False,
                 outdir=None,
                 txt=None,
                 calcstress=False,
                 smearing='fd',
                 sigma=0.1,
                 fix_magmom=False,
                 isolated=None,
                 U=None,
                 J=None,
                 U_alpha=None,
                 U_projection_type='atomic',
                 nqx1=None,
                 nqx2=None,
                 nqx3=None,
                 exx_fraction=None,
                 screening_parameter=None,
                 exxdiv_treatment=None,
                 ecutvcut=None,
                 tot_magnetization=-1,
                 occupations='smearing',
                 dipole={'status': False},
                 field={'status': False},
                 output={'disk_io': 'default',
                         'avoidio': False,
                         'removewf': True,
                         'removesave': False,
                         'wf_collect': False},
                 convergence={'energy': 1e-6,
                              'mixing': 0.7,
                              'maxsteps': 100,
                              'diag': 'david'},
                 startingpot=None,
                 startingwfc=None,
                 ion_positions=None,
                 parflags=None,
                 newforcearray=True,
                 verbose='low',
                 # automatically generated list of parameters
                 # some coincide with ase-style names
                 iprint=None,
                 tstress=None,
                 tprnfor=None,
                 dt=None,
                 lkpoint_dir=None,
                 max_seconds=None,
                 etot_conv_thr=None,
                 forc_conv_thr=None,
                 tefield=None,
                 dipfield=None,
                 lelfield=None,
                 nberrycyc=None,
                 lorbm=None,
                 lberry=None,
                 gdir=None,
                 nppstr=None,
                 nbnd=None,
                 ecutwfc=None,
                 ecutrho=None,
                 ecutfock=None,
                 force_symmorphic=None,
                 use_all_frac=None,
                 one_atom_occupations=None,
                 starting_spin_angle=None,
                 degauss=None,
                 nspin=None,
                 ecfixed=None,
                 qcutz=None,
                 q2sigma=None,
                 x_gamma_extrapolation=None,
                 lda_plus_u=None,
                 lda_plus_u_kind=None,
                 edir=None,
                 emaxpos=None,
                 eopreg=None,
                 eamp=None,
                 clambda=None,
                 report=None,
                 lspinorb=None,
                 esm_w=None,
                 esm_efield=None,
                 esm_nfit=None,
                 london=None,
                 london_s6=None,
                 london_rcut=None,
                 xdm=None,
                 xdm_a1=None,
                 xdm_a2=None,
                 electron_maxstep=None,
                 scf_must_converge=None,
                 conv_thr=None,
                 adaptive_thr=None,
                 conv_thr_init=None,
                 conv_thr_multi=None,
                 mixing_beta=None,
                 mixing_ndim=None,
                 mixing_fixed_ns=None,
                 ortho_para=None,
                 diago_thr_init=None,
                 diago_cg_maxiter=None,
                 diago_david_ndim=None,
                 diago_full_acc=None,
                 efield=None,
                 tqr=None,
                 remove_rigid_rot=None,
                 tempw=None,
                 tolp=None,
                 delta_t=None,
                 nraise=None,
                 refold_pos=None,
                 upscale=None,
                 bfgs_ndim=None,
                 trust_radius_max=None,
                 trust_radius_min=None,
                 trust_radius_ini=None,
                 w_1=None,
                 w_2=None,
                 wmass=None,
                 press_conv_thr=None,
                 site=None,
                 force_consistent=False,)

        if charge is None:
            self.defaults['tot_charge'] = tot_charge
        else:
            self.defaults['tot_charge'] = charge


        # internal attributes

        self._initialized = False
        self._running = False

        self.results = {}

        # give original espresso style input names
        # preference over ase / dacapo - style names
        if ecutwfc is not None:
            self.pw = ecutwfc
        if ecutrho is not None:
            self.dw = ecutwfc
        if nbnd is not None:
            self.nbands = nbnd

        self.site = site

        # Variables that cannot be set by inputs
        self.nvalence = None
        self.nel = None
        self.fermi_input = False
        # Auto create variables from input
        self.input_update()

        if atoms is not None:
            atoms.set_calculator(self)

        self.parameters = dict()

    @property
    def site(self):
        return self._site

    @site.setter
    def site(self, value):
        if value is None:
            self._site = SiteConfig.check_scheduler()
        else:
            self._site = value

    @property
    def name(self):
        return self.get_name()

    def get_name(self):
        return 'QE-ASE interface'

    @property
    def kpts(self):
        return self._kpts

    @kpts.setter
    def kpts(self, value):
        '''
        Resolve the `kpts` and store as an attribute
        '''

        if isinstance(value, str):

            if value.lower() == 'gamma':
                self._kpts = value.lower()
            else:
                raise ValueError('wrong <kpts>: {}'.format(value))

        elif isinstance(value, (float, int)):

            if self.atoms is not None:
                from ase.calculators.calculator import kptdensity2monkhorstpack
                self._kpts = kptdensity2monkhorstpack(self.atoms, value)
            else:
                raise ValueError('cannot calculate the Monkhorst-Pack grid '
                                 'without the <atoms> object specified')
        elif isinstance(value, (list, tuple)):

            self._kpts = np.asarray(value)

        elif type(value).__module__ == np.__name__:

            self._kpts = value

        else:
            raise ValueError('unknown <kpts> type: {}'.format(type(value)))

    def get_version(self):
        return __version__

    def set_pseudo_path(self):
        '''
        Check if the pseudopotential path is defined and set it from
        the environmental variable `ESP_PSP_PATH` if necessary
        '''

        if self.psppath is None:
            if os.environ['ESP_PSP_PATH'] is not None:
                self.psppath = os.environ['ESP_PSP_PATH']
            else:
                raise ValueError('Unable to find pseudopotential path.'
                    'Consider setting <ESP_PSP_PATH> environment variable')

    def initialize(self, atoms):
        '''
        Create the scratch directories and pw.inp input file and
        prepare for writing the input file
        '''

        if not self._initialized:
            self.create_outdir()
            if self.site.usehostfile:
                self.site.write_local_hostfile()

        self.set_pseudo_path()
        self.atoms = atoms.copy()
        self.natoms = len(self.atoms)
        self.atoms2species()

        self.check_spinpol()
        self._initialized = True

    def calculate(self, atoms, properties=['energy']):
        '''
        Run the calculation and retrieve the results
        '''

        if atoms is not None:
            self.atoms = atoms.copy()

        # initialize
        self.initialize(atoms)

        # write input
        self.write_input()

        # run
        self.run()

        self.recalculate = True

        # check for errors
        self.checkerror()

        # parse results
        self.read()

        self.set_results(atoms)

    def set_atoms(self, atoms):

        if self.atoms is None or not self.started:
            self.atoms = atoms.copy()
        else:
            if len(atoms) != len(self.atoms):
                self.stop()
                self.nvalence = None
                self.nel = None
                self.recalculate = True

            x = atoms.cell - self.atoms.cell
            if np.max(x) > 1E-13 or np.min(x) < -1E-13:
                self.stop()
                self.recalculate = True
            if (atoms.get_atomic_numbers()!=self.atoms.get_atomic_numbers()).any():
                self.stop()
                self.nvalence = None
                self.nel = None
                self.recalculate = True
            x = atoms.positions - self.atoms.positions
            if np.max(x) > 1E-13 or np.min(x) < -1E-13 or (not self.started and not self.got_energy):
                self.recalculate = True
        self.atoms = atoms.copy()

    def update(self, atoms):
        '''
        Check if the atoms object has changes and perform a calcualtion
        when it does
        '''

        if self.atoms is None:
            self.set_atoms(atoms)

        if self.calculation_required(atoms, ['energy']):
            self.calculate(atoms)
            self.recalculate = False

    def set_results(self, atoms):
        '''
        Read the results from the file and populate the `results` dictionary
        '''

        self.results = {'energy': self.read_energies()[0],
                        'forces': self.read_forces(),
                        'stress': self.read_stress(),
                        'absolute magnetization': self.read_magnetizations()[0],
                        'total magnetization': self.read_magnetizations()[1],
                        #'dipole': self.get_dipole(),
                        #'charges': self.get_charges(),
                        #'magmom': 0.0,
                        #'magmoms': np.zeros(len(atoms))
                        }

    def read(self):
        '''
        Read the output file and set the attributes
        '''

        self.energy_zero, self.energy_free = self.read_energies(getall=False)
        self.forces = self.read_forces()
        self.stress = self.read_stress()
        positions = self.read_positions()
        cell = self.read_cell()

    def input_update(self):
        '''
        Run initialization functions, such that this can be called if variables
        in espresso are changes using set or directly.
        '''

        if self.dipole is None:
            self.dipole = {'status': False}
        if self.field is None:
            self.field = {'status': False}

        if self.convergence is None:
            self.conv_thr = 1.0e-6 / Rydberg
        else:
            if 'energy' in list(self.convergence.keys()):
                self.conv_thr = self.convergence['energy'] / Rydberg
            else:
                self.conv_thr = 1.0e-6 / Rydberg

        if self.beefensemble:
            if self.xc.upper().find('BEEF') < 0:
                raise KeyError("ensemble-energies only work with xc=BEEF or variants of it!")

        self.started = False
        self.got_energy = False

    def create_outdir(self):
        '''
        Create the necessary directory structure to run the calculation and
        assign file names
        '''


        self.localtmp = self.site.make_localtmp(self.outdir)
        self.scratch = self.site.make_scratch()

        if self.txt is None:
            self.log = self.localtmp.joinpath('log')
        else:
            self.log = self.localtmp.joinpath(self.txt)

        atexit.register(self.clean)

    def set(self, **kwargs):
        '''
        Define settings for the Quantum Espresso calculator object after it
        has been initialized. This is done in the following way::

        Example:

            ::

              >>> calc = Espresso(...)
              >>> atoms = set.calculator(calc)
              >>> calc.set(xc='BEEF')

        Warning:

            No input validation is done

        '''

        for key, value in list(kwargs.items()):
            setattr(self, key, value)

        self._initialized = False
        self.input_update()
        self.recalculate = True

    @preserve_cwd
    def run(self):
        '''
        Execute the expresso program `pw.x`
        '''

        if self.site.batchmode:
            self.localtmp.chdir()
            Path.copy2(self.localtmp.joinpath('pw.inp'), self.scratch)

            if self.calculation != 'hund':

                command = self.site.get_proc_mpi_command(self.scratch,
                                'pw.x ' + self.parflags + ' -in pw.inp')

                if self.ion_dynamics == 'ase3':
                    raise ValueError('use interactive version <iEspresso> for ion_dynamics="ase3"')
                else:
                    with open(self.log, 'ab') as flog:
                        flog.write(self.get_output_header().encode('utf-8'))
                        exitcode = subprocess.call(command, stdout=flog)
                    if exitcode != 0:
                        raise RuntimeError('something went wrong:', exitcode)

            else:  # calculation == 'hund'
                self.site.runonly_perProcMpiExec(self.scratch,' pw.x -in pw.inp >>'+self.log)
                os.system("sed s/occupations.*/occupations=\\'fixed\\',/ <"+self.localtmp+"/pw.inp | sed s/ELECTRONS/ELECTRONS\\\\n\ \ startingwfc=\\'file\\',\\\\n\ \ startingpot=\\'file\\',/ | sed s/conv_thr.*/conv_thr="+num2str(self.conv_thr)+",/ | sed s/tot_magnetization.*/tot_magnetization="+num2str(self.totmag)+",/ >"+self.localtmp+"/pw2.inp")
                os.system(self.site.perHostMpiExec+' cp '+self.localtmp+'/pw2.inp '+self.scratch)
                self.cinp, self.cout = self.site.do_perProcMpiExec(self.scratch,'pw.x '+self.parflags+' -in pw2.inp')

        else:  # not in batchmode

            pwinp = self.localtmp.joinpath('pw.inp')
            Path.copy(pwinp, self.scratch)
            command = ['pw.x', '-in', 'pw.inp']
            if self.calculation != 'hund':
                self.scratch.chdir()
                with open(self.log, 'ab') as flog:
                    flog.write(self.get_output_header().encode('utf-8'))
                    exitcode = subprocess.call(command, stdout=flog)

            else:
                self.scratch.chdir()
                subprocess.call('pw.x -in pw.inp >> ' + self.log, shell=True)

                os.system("sed s/occupations.*/occupations=\\'fixed\\',/ <"+self.localtmp+"/pw.inp | sed s/ELECTRONS/ELECTRONS\\\\n\ \ startingwfc=\\'file\\',\\\\n\ \ startingpot=\\'file\\',/ | sed s/conv_thr.*/conv_thr="+num2str(self.conv_thr)+",/ | sed s/tot_magnetization.*/tot_magnetization="+num2str(self.totmag)+",/ >"+self.localtmp+"/pw2.inp")
                shutil.copy(os.path.join(self.localtmp, 'pw2.inp'), self.scratch)
                self.scratch.chdir()
                self.cinp, self.cout = os.popen2('pw.x -in pw2.inp')

            self._running = True

    def stop(self):
        if self._running:

            self._running = False

    def clean(self):
        '''
        Remove the temporary files and directories
        '''

        os.chdir(self.site.submitdir)

        try:
            self.stop()
        except:
            pass

        if self.output is not None:
            if 'removewf' in list(self.output.keys()):
                removewf = self.output['removewf']
            else:
                removewf = True
            if 'removesave' in list(self.output.keys()):
                removesave = self.output['removesave']
            else:
                removesave = False
        else:
            removewf = True
            removesave = False

        toremove = ['*.wfc*', '*.update', '*.igk*', '*.hub']
        if removewf:
            for pattern in toremove:
                for fil in self.scratch.files(pattern):
                    fil.remove()

        if not removesave:
            Path.copytree(self.scratch, self.localtmp.joinpath(self.scratch.basename()))

        self.scratch.rmtree_p()

        if hasattr(self.site, 'mpdshutdown') and 'QEASE_MPD_ISSHUTDOWN' not in list(os.environ.keys()):
            os.environ['QEASE_MPD_ISSHUTDOWN'] = 'yes'
            os.system(self.site.mpdshutdown)

    def atoms2species(self):
        '''
        Define several properties of the quantum espresso species from the ase
        atoms object. Takes into account that different spins (or different U
        etc.) on same kind of chemical elements are considered different
        species in quantum espresso
        '''

        symbols = self.atoms.get_chemical_symbols()
        masses = np.nan_to_num(self.atoms.get_masses())
        magmoms = list(self.atoms.get_initial_magnetic_moments())
        if len(magmoms) < len(symbols):
            magmoms += list(np.zeros(len(symbols) - len(magmoms), np.float))
        pos = self.atoms.get_scaled_positions()

        if self.U is not None:
            if isinstance(self.U, dict):
                Ulist = np.zeros(len(symbols), np.float)
                for i, s in enumerate(symbols):
                    if s in list(self.U.keys()):
                        Ulist[i] = self.U[s]
            else:
                Ulist = list(self.U)
                if len(Ulist) < len(symbols):
                    Ulist += list(np.zeros(len(symbols) - len(Ulist), np.float))
        else:
            Ulist = np.zeros(len(symbols), np.float)

        if self.J is not None:
            if isinstance(self.J, dict):
                Jlist = np.zeros(len(symbols), np.float)
                for i, s in enumerate(symbols):
                    if s in list(self.J.keys()):
                        Jlist[i] = self.J[s]
            else:
                Jlist = list(self.J)
                if len(Jlist) < len(symbols):
                    Jlist += list(np.zeros(len(symbols) - len(Jlist), np.float))
        else:
            Jlist = np.zeros(len(symbols), np.float)

        if self.U_alpha is not None:
            if isinstance(self.U_alpha, dict):
                U_alphalist = np.zeros(len(symbols), np.float)
                for i, s in enumerate(symbols):
                    if s in list(self.U_alpha.keys()):
                        U_alphalist[i] = self.U_alpha[s]
            else:
                U_alphalist = list(self.U_alpha)
                if len(U_alphalist) < len(symbols):
                    U_alphalist += list(np.zeros(len(symbols) - len(U_alphalist), np.float))
        else:
            U_alphalist = np.zeros(len(symbols), np.float)

        self.species = []
        self.specprops = []
        dic = {}
        symcounter = {}
        for s in symbols:
            symcounter[s] = 0
        for i in range(len(symbols)):
            key = symbols[i] + '_m%.14eU%.14eJ%.14eUa%.14e' % (magmoms[i], Ulist[i], Jlist[i], U_alphalist[i])
            if key in list(dic.keys()):
                self.specprops.append((dic[key][1], pos[i]))
            else:
                symcounter[symbols[i]] += 1
                spec = symbols[i] + newstr(symcounter[symbols[i]])
                dic[key] = [i, spec]
                self.species.append(spec)
                self.specprops.append((spec, pos[i]))

        self.nspecies = len(self.species)
        self.specdict = {}
        for i, s in list(dic.values()):
            self.specdict[s] = speciestuple(s.strip('0123456789'),  # chemical symbol w/o index
                                            masses[i],
                                            magmoms[i],
                                            Ulist[i],
                                            Jlist[i],
                                            U_alphalist[i])

    def parse_upf(self, symbols, fext='.UPF'):
        '''
        Parse the `<PP_HEADER>` section from the pseudopotential files in the
        Unified Pseudopotential Format (UPF)

        Args:
            symbols (`list` of str) :
                List of symbols for which the UPF files will be parsed
            fext (str) :
                File extension of the pseudopotential files, defaults to `.UPF`

        Returns:
            parsed (`dict`) :
                Dictionary with filed parsed from the `<PP_HEADER>` section

        See also:
           `<http://www.quantum-espresso.org/pseudopotentials/unified-pseudopotential-format/>`_

        '''

        pat = re.compile(r'(?P<block><PP_HEADER)\s+(?P<entries>.*?)/>',
                         flags=re.DOTALL | re.IGNORECASE)

        parsed = OrderedDict()
        for symbol in symbols:
            parsed[symbol] = dict()
            fname = os.path.join(self.psppath, symbol + fext)
            if os.path.exists(fname):
                with open(fname, 'r') as pspfile:
                    match = pat.search(pspfile.read())
                if match:
                    for line in [s.strip() for s in match.group("entries").split("\n")]:
                        key, value = line.split("=")
                        parsed[symbol][key] = value.strip('"')
            else:
                raise IOError('Missing pseudopotential file: {}'.format(fname))
        return parsed

    def get_nvalence(self, fext='.UPF'):
        '''
        Get the number of valence electrons from pseudopotential or paw setup
        '''

        symbols = [self.specdict[x].symbol for x in self.species]
        pattern = re.compile('z\s*_?valence', re.IGNORECASE)
        nel = dict()
        for symbol in symbols:
            fname = os.path.join(self.psppath, symbol + fext)
            with open(fname, 'r') as fupf:
                for line in fupf:
                    if pattern.search(line):
                        if '=' in line:
                            name, value = line.split('=')
                            nel[symbol] = int(round(float(value.strip('"\n'))))
                        else:
                            lsplit = line.split()
                            nel[symbol] = int(round(float(lsplit[0])))

        nvalence = np.zeros(len(self.specprops), np.int)
        for i, x in enumerate(self.specprops):
            nvalence[i] = nel[self.specdict[x[0]].symbol]
        return nvalence, nel

    def get_number_of_scf_steps(self, all=False):
        """Get number of steps for convered scf. Returns an array.
        Option 'all' gives all numbers of steps in log,
        not only for the latest scf."""
        if all:
            tail = 'tail'
        else:
            tail = 'tail -1'
        p = os.popen('grep "convergence has been achieved in" '+self.log+' | '+tail, 'r')
        s = p.readlines()
        p.close()
        if not all:
            assert len(s) < 2
        if len(s) == 0:
            return None
        else:
            out = []
            for s_ in s:
                tmp = s_.split('in')
                out.append(int(tmp[-1].split('iterations')[0]))
            return out

    def get_number_of_bfgs_steps(self):
        """Get total number of internal BFGS steps."""
        p = os.popen('grep "bfgs converged in" '+self.log+' | tail -1', 'r')
        s = p.readlines()
        p.close()
        assert len(s) < 2
        if len(s) == 0:
            return None
        else:
            tmp = s[0].split('and')
            return int(tmp[-1].split('bfgs')[0])

    def get_forces(self, atoms):
        self.update(atoms)
        if self.newforcearray:
            return self.forces.copy()
        else:
            return self.forces

    def write_input(self, inputname='pw.inp', calculation=None,
                    overridekpts=None, overridekptshift=None,
                    overridenbands=None, suppressforcecalc=False,
                    usetetrahedra=False):

        if self.atoms is None:
            raise ValueError('no atoms defined')

        fname = self.localtmp.joinpath(inputname)
        
        finp = open(fname, 'w')

        # &CONTROL
        if calculation is None:
            if self.calculation.lower() == 'hund':
                print('&CONTROL\n  calculation=\'scf\',\n  prefix=\'calc\',', file=finp)
            else:
                print('&CONTROL\n  calculation=\'' + self.calculation + '\',\n  prefix=\'calc\',', file=finp)
            ionssec = self.calculation not in ('scf', 'nscf', 'bands', 'hund')
        else:
            print('&CONTROL\n  calculation=\'' + calculation + '\',\n  prefix=\'calc\',', file=finp)
            ionssec = calculation not in ('scf', 'nscf', 'bands', 'hund')

        if self.nstep is not None:
            print('  nstep=' + str(self.nstep) + ',', file=finp)

        if self.verbose != 'low':
            print('  verbosity=\'' + self.verbose + '\',', file=finp)

        print('  pseudo_dir=\'' + self.psppath + '\',', file=finp)
        print('  outdir=\'.\',', file=finp)
        efield = (self.field['status'] is True)
        dipfield = (self.dipole['status'] is True)
        if efield or dipfield:
            print('  tefield=.true.,', file=finp)
            if dipfield:
                print('  dipfield=.true.,', file=finp)
        if not self.dontcalcforces and not suppressforcecalc:
            print('  tprnfor=.true.,', file=finp)
            if self.calcstress:
                print('  tstress=.true.,', file=finp)
            if self.output is not None:
                if 'avoidio' in list(self.output.keys()):
                    if self.output['avoidio']:
                        self.output['disk_io'] = 'none'
                if 'disk_io' in list(self.output.keys()):
                    if self.output['disk_io'] in ['high', 'low', 'none']:
                        print('  disk_io=\'' + self.output['disk_io'] + '\',',
                              file=finp)

                if 'wf_collect' in list(self.output.keys()):
                    if self.output['wf_collect']:
                        print('  wf_collect=.true.,', file=finp)
        if self.ion_dynamics != 'ase3':
            # we basically ignore convergence of total energy differences
            # between ionic steps and only consider fmax as in ase
            print('  etot_conv_thr=1d0,', file=finp)
            print('  forc_conv_thr=' + num2str(self.fmax / (Rydberg / Bohr)) +
                  ',', file=finp)

        # turn on fifo communication if espsite.py is set up that way
        if hasattr(self.site, 'fifo'):
            if self.site.fifo:
                print('  ase_fifo=.true.,', file=finp)

        # automatically generated parameters
        if self.iprint is not None:
            print('  iprint=' + str(self.iprint) + ',', file=finp)
        if self.tstress is not None:
            print('  tstress=' + bool2str(self.tstress) + ',', file=finp)
        if self.tprnfor is not None:
            print('  tprnfor=' + bool2str(self.tprnfor) + ',', file=finp)
        if self.dt is not None:
            print('  dt='+num2str(self.dt)+',', file=finp)
        if self.lkpoint_dir is not None:
            print('  lkpoint_dir='+bool2str(self.lkpoint_dir)+',', file=finp)
        if self.max_seconds is not None:
            print('  max_seconds='+num2str(self.max_seconds)+',', file=finp)
        if self.etot_conv_thr is not None:
            print('  etot_conv_thr='+num2str(self.etot_conv_thr)+',', file=finp)
        if self.forc_conv_thr is not None:
            print('  forc_conv_thr='+num2str(self.forc_conv_thr)+',', file=finp)
        if self.tefield is not None:
            print('  tefield='+bool2str(self.tefield)+',', file=finp)
        if self.dipfield is not None:
            print('  dipfield='+bool2str(self.dipfield)+',', file=finp)
        if self.lelfield is not None:
            print('  lelfield='+bool2str(self.lelfield)+',', file=finp)
        if self.nberrycyc is not None:
            print('  nberrycyc='+str(self.nberrycyc)+',', file=finp)
        if self.lorbm is not None:
            print('  lorbm='+bool2str(self.lorbm)+',', file=finp)
        if self.lberry is not None:
            print('  lberry='+bool2str(self.lberry)+',', file=finp)
        if self.gdir is not None:
            print('  gdir='+str(self.gdir)+',', file=finp)
        if self.nppstr is not None:
            print('  nppstr='+str(self.nppstr)+',', file=finp)


        ### &SYSTEM ###
        print('/\n&SYSTEM\n  ibrav=0,\n  celldm(1)=1.8897261245650618d0,', file=finp)
        print('  nat='+str(self.natoms)+',', file=finp)
        self.atoms2species()  # self.convertmag2species()
        print('  ntyp='+str(self.nspecies)+',', file=finp)
        if self.tot_charge is not None:
            print('  tot_charge='+num2str(self.tot_charge)+',', file=finp)
        if self.calculation != 'hund':
            inimagscale = 1.0
        else:
            inimagscale = 0.9
        if self.fix_magmom:
            assert self.spinpol
            self.totmag = self.summed_magmoms
            print('  tot_magnetization='+num2str(self.totmag*inimagscale)+',', file=finp)
        elif self.tot_magnetization != -1:
            if self.tot_magnetization != 'hund':
                self.totmag = self.tot_magnetization
            else:
                from .atomic_configs import hundmag
                self.totmag = sum([hundmag(x) for x in self.atoms.get_chemical_symbols()])
            print('  tot_magnetization='+num2str(self.totmag*inimagscale)+',', file=finp)
        print('  ecutwfc=' + num2str(self.pw / Rydberg) + ',', file=finp)
        print('  ecutrho=' + num2str(self.dw / Rydberg) + ',', file=finp)
        if self.fw is not None:
            print('  ecutfock=' + num2str(self.fw / Rydberg) + ',', file=finp)
        #temporarily (and optionally) change number of bands for nscf calc.
        if overridenbands is not None:
            if self.nbands is None:
                nbandssave = None
            else:
                nbandssave = self.nbands
            self.nbands = overridenbands
        if self.nbands is not None:
            # set number of bands
            if self.nbands > 0:
                self.nbnd = int(self.nbands)
            else:
            # if self.nbands is negative create - self.nbands extra bands
                if self.nvalence is None:
                    self.nvalence, self.nel = self.get_nvalence()
                if self.noncollinear:
                    self.nbnd = int(np.sum(self.nvalence) - self.nbands * 2.0)
                else:
                    self.nbnd = int(np.sum(self.nvalence) / 2.0 - self.nbands)
            print('  nbnd=' + str(self.nbnd) + ',', file=finp)
        if overridenbands is not None:
            self.nbands = nbandssave
        if usetetrahedra:
            print('  occupations=\'tetrahedra\',', file=finp)
        else:
            if abs(self.sigma) > 1e-13:
                print('  occupations=\'' + self.occupations + '\',', file=finp)
                print('  smearing=\'' + self.smearing + '\',', file=finp)
                print('  degauss=' + num2str(self.sigma / Rydberg) + ',', file=finp)
            else:
                if self.spinpol:
                    assert self.fix_magmom
                print('  occupations=\'fixed\',', file=finp)
        if self.spinpol:
            print('  nspin=2,', file=finp)
            spcount = 1
            if self.nel is None:
                self.nvalence, self.nel = self.get_nvalence()
            for species in self.species:  # FOLLOW SAME ORDERING ROUTINE AS FOR PSP
                spec = self.specdict[species]
                el = spec.symbol
                mag = spec.magmom / self.nel[el]
                assert np.abs(mag) <= 1.0  # magnetization oversaturated!!!
                print('  starting_magnetization(%d)=%s,' % (spcount, num2str(float(mag))), file=finp)
                spcount += 1
        elif self.noncollinear:
            print('  noncolin=.true.,', file=finp)
            if self.spinorbit:
                print('  lspinorb=.true.', file=finp)
            spcount = 1
            if self.nel is None:
                self.nvalence, self.nel = self.get_nvalence()
            for species in self.species:  # FOLLOW SAME ORDERING ROUTINE AS FOR PSP
                spec = self.specdict[species]
                el = spec.symbol
                mag = spec.magmom / self.nel[el]
                assert np.abs(mag) <= 1.0  # magnetization oversaturated!!!
                print('  starting_magnetization(%d)=%s,' % (spcount, num2str(float(mag))), file=finp)
                spcount += 1
        if self.isolated is not None:
            print('  assume_isolated=\'' + self.isolated + '\',', file=finp)
        print('  input_dft=\'' + self.xc + '\',', file=finp)
        if self.beefensemble:
            print('  ensemble_energies=.true.,', file=finp)
            if self.printensemble:
                print('  print_ensemble_energies=.true.,', file=finp)
            else:
                print('  print_ensemble_energies=.false.,', file=finp)
        edir = 3
        if dipfield:
            try:
                edir = self.dipole['edir']
            except:
                pass
        elif efield:
            try:
                edir = self.field['edir']
            except:
                pass
        if dipfield or efield:
            print('  edir=' + str(edir) + ',', file=finp)
        if dipfield:
            if 'emaxpos' in list(self.dipole.keys()):
                emaxpos = self.dipole['emaxpos']
            else:
                emaxpos = self.find_max_empty_space(edir)
            if 'eopreg' in list(self.dipole.keys()):
                eopreg = self.dipole['eopreg']
            else:
                eopreg = 0.025
            if 'eamp' in list(self.dipole.keys()):
                eamp = self.dipole['eamp']
            else:
                eamp = 0.0
            print('  emaxpos='+num2str(emaxpos)+',', file=finp)
            print('  eopreg='+num2str(eopreg)+',', file=finp)
            print('  eamp='+num2str(eamp)+',', file=finp)
        if efield:
            if 'emaxpos' in list(self.field.keys()):
                emaxpos = self.field['emaxpos']
            else:
                emaxpos = 0.0
            if 'eopreg' in list(self.field.keys()):
                eopreg = self.field['eopreg']
            else:
                eopreg = 0.0
            if 'eamp' in list(self.field.keys()):
                eamp = self.field['eamp']
            else:
                eamp = 0.0
            print('  emaxpos='+num2str(emaxpos)+',', file=finp)
            print('  eopreg='+num2str(eopreg)+',', file=finp)
            print('  eamp='+num2str(eamp)+',', file=finp)
        if self.U is not None or self.J is not None or self.U_alpha is not None:
            print('  lda_plus_u=.true.,', file=finp)
            if self.J is not None:
                print('  lda_plus_u_kind=1,', file=finp)
            else:
                print('  lda_plus_u_kind=0,', file=finp)
            print('  U_projection_type=\"%s\",' % (self.U_projection_type), file=finp)
            if self.U is not None:
                for i, s in enumerate(self.species):
                    spec = self.specdict[s]
                    el = spec.symbol
                    Ui = spec.U
                    print('  Hubbard_U('+str(i+1)+')='+num2str(Ui)+',', file=finp)
            if self.J is not None:
                for i, s in enumerate(self.species):
                    spec = self.specdict[s]
                    el = spec.symbol
                    Ji = spec.J
                    print('  Hubbard_J(1,'+str(i+1)+')='+num2str(Ji)+',', file=finp)
            if self.U_alpha is not None:
                for i, s in enumerate(self.species):
                    spec = self.specdict[s]
                    el = spec.symbol
                    U_alphai = spec.U_alpha
                    print('  Hubbard_alpha('+str(i+1)+')='+num2str(U_alphai)+',', file=finp)

        sys_int_attrs = ['nqx1', 'nqx2', 'nqx3', 'nspin', 'lda_plus_u_kind', 'edir', 'report',
                         'esm_nfit']
        for attr in sys_int_attrs:
            value = getattr(self, attr)
            if value is not None:
                print('  {0:s}={1:d},'.format(attr, value), file=finp)

        sys_float_attrs = ['exx_fraction', 'screening_parameter', 'ecutvcut', 'ecutfock',
                           'degauss', 'ecfixed', 'qcutz', 'q2sigma', 'emaxpos', 'eopreg', 'eamp',
                           'esm_w', 'esm_efield', 'london_s6', 'london_rcut', 'xdm_a1', 'xdm_a2']
        for attr in sys_float_attrs:
            value = getattr(self, attr)
            if value is not None:
                print('  {0:s}={1:s},'.format(attr, num2str(value)), file=finp)

        sys_str_attrs = ['exxdiv_treatment']
        for attr in sys_str_attrs:
            value = getattr(self, attr)
            if value is not None:
                print('  {0:s}={1:s},'.format(attr, value), file=finp)

        sys_bool_attrs = ['nosym', 'noinv', 'nosym_evc', 'no_t_rev', 'force_symmorphic',
                          'use_all_frac', 'one_atom_occupations', 'starting_spin_angle',
                          'x_gamma_extrapolation', 'lda_plus_u', 'lspinorb', 'london', 'xdm']
        for attr in sys_bool_attrs:
            value = getattr(self, attr)
            if value is not None:
                print('  {0:s}={1:s},'.format(attr, bool2str(value)), file=finp)

        if self.clambda is not None:
            print('  lambda={0:s},'.format(num2str(self.clambda)), file=finp)

        if self.fft_grid is not None:  #RK
            print('  nr1={0:d},'.format(self.fft_grid[0]), file=finp)
            print('  nr2={0:d},'.format(self.fft_grid[1]), file=finp)
            print('  nr3={0:d},'.format(self.fft_grid[2]), file=finp)

        ### &ELECTRONS ###
        print('/\n&ELECTRONS', file=finp)
        if 'diag' in list(self.convergence.keys()):
            print("  diagonalization='{0:s}',".format(self.convergence['diag']), file=finp)

        if self.calculation != 'hund':
            print('  conv_thr=' + num2str(self.conv_thr) + ',', file=finp)
        else:
            print('  conv_thr=' + num2str(self.conv_thr*500.) + ',', file=finp)
        for x in list(self.convergence.keys()):
            if x == 'mixing':
                print('  mixing_beta='+num2str(self.convergence[x])+',', file=finp)
            elif x == 'maxsteps':
                print('  electron_maxstep='+str(self.convergence[x])+',', file=finp)
            elif x == 'nmix':
                print('  mixing_ndim='+str(self.convergence[x])+',', file=finp)
            elif x == 'mixing_mode':
                print('  mixing_mode=\''+self.convergence[x]+'\',', file=finp)
            elif x == 'diago_cg_maxiter':
                print('  diago_cg_maxiter='+str(self.convergence[x])+',', file=finp)
        if self.startingpot is not None and self.calculation != 'hund':
            print('  startingpot=\''+self.startingpot+'\',', file=finp)
        if self.startingwfc is not None and self.calculation != 'hund':
            print('  startingwfc=\''+self.startingwfc+'\',', file=finp)

        # automatically generated parameters
        el_int_attrs = ['electron_maxstep', 'mixing_ndim', 'mixing_fixed_ns',
                        'ortho_para', 'diago_cg_maxiter', 'diago_david_ndim']
        for attr in el_int_attrs:
            value = getattr(self, attr)
            if value is not None:
                print('  {0:s}={1:d},'.format(attr, value), file=finp)

        el_float_attrs = ['conv_thr_init', 'conv_thr_multi', 'mixing_beta',
                        'diago_thr_init', 'efield']
        for attr in el_float_attrs:
            value = getattr(self, attr)
            if value is not None:
                print('  {0:s}={1:s},'.format(attr, num2str(value)), file=finp)

        el_bool_attrs = ['scf_must_converge', 'adaptive_thr', 'diago_full_acc', 'tqr']
        for attr in el_bool_attrs:
            value = getattr(self, attr)
            if value is not None:
                print('  {0:s}={1:s},'.format(attr, bool2str(value)), file=finp)

        ### &IONS ###
        if self.ion_dynamics == 'ase3' or not ionssec:
            simpleconstr, otherconstr = [], []
        else:
            simpleconstr, otherconstr = convert_constraints(self.atoms)

        if self.ion_dynamics is None:
            self.optdamp = False
        else:
            self.optdamp = (self.ion_dynamics.upper() == 'DAMP')

        if self.ion_dynamics is not None and ionssec:
            if len(otherconstr) != 0:
                print('/\n&IONS\n  ion_dynamics=\'damp\',', file=finp)
                self.optdamp = True
            else:
                print('/\n&IONS\n  ion_dynamics=\'' + self.ion_dynamics + '\',', file=finp)
            if self.ion_positions is not None:
                print('  ion_positions=\''+self.ion_positions+'\',', file=finp)
        elif self.ion_positions is not None:
            print('/\n&IONS\n  ion_positions=\''+self.ion_positions+'\',', file=finp)

        # automatically generated parameters
        if self.remove_rigid_rot is not None:
            print('  remove_rigid_rot='+bool2str(self.remove_rigid_rot)+',', file=finp)
        if self.tempw is not None:
            print('  tempw='+num2str(self.tempw)+',', file=finp)
        if self.tolp is not None:
            print('  tolp='+num2str(self.tolp)+',', file=finp)
        if self.delta_t is not None:
            print('  delta_t='+num2str(self.delta_t)+',', file=finp)
        if self.nraise is not None:
            print('  nraise='+str(self.nraise)+',', file=finp)
        if self.refold_pos is not None:
            print('  refold_pos='+bool2str(self.refold_pos)+',', file=finp)
        if self.upscale is not None:
            print('  upscale='+num2str(self.upscale)+',', file=finp)
        if self.bfgs_ndim is not None:
            print('  bfgs_ndim='+str(self.bfgs_ndim)+',', file=finp)
        if self.trust_radius_max is not None:
            print('  trust_radius_max='+num2str(self.trust_radius_max)+',', file=finp)
        if self.trust_radius_min is not None:
            print('  trust_radius_min='+num2str(self.trust_radius_min)+',', file=finp)
        if self.trust_radius_ini is not None:
            print('  trust_radius_ini='+num2str(self.trust_radius_ini)+',', file=finp)
        if self.w_1 is not None:
            print('  w_1='+num2str(self.w_1)+',', file=finp)
        if self.w_2 is not None:
            print('  w_2='+num2str(self.w_2)+',', file=finp)

        ### &CELL ###
        if self.cell_dynamics is not None:
            print('/\n&CELL\n  cell_dynamics=\''+self.cell_dynamics+'\',', file=finp)
            if self.press is not None:
                print('  press='+num2str(self.press)+',', file=finp)
            if self.dpress is not None:
                print('  press_conv_thr='+num2str(self.dpress)+',', file=finp)
            if self.cell_factor is not None:
                print('  cell_factor='+num2str(self.cell_factor)+',', file=finp)
            if self.cell_dofree is not None:
                print('  cell_dofree=\''+self.cell_dofree+'\',', file=finp)

        # automatically generated parameters
        if self.wmass is not None:
            print('  wmass='+num2str(self.wmass)+',', file=finp)
        if self.press_conv_thr is not None:
            print('  press_conv_thr='+num2str(self.press_conv_thr)+',', file=finp)


        ### CELL_PARAMETERS
        print('/\nCELL_PARAMETERS alat', file=finp)
        for i in range(3):
            print('%21.15fd0 %21.15fd0 %21.15fd0' % tuple(self.atoms.cell[i]), file=finp)

        print('ATOMIC_SPECIES', file=finp)
        for species in self.species:   # PSP ORDERING FOLLOWS SPECIESINDEX
            spec = self.specdict[species]
            print(species, num2str(spec.mass), spec.symbol + '.UPF', file=finp)

        print('ATOMIC_POSITIONS {crystal}', file=finp)
        if len(simpleconstr) == 0:
            for species, pos in self.specprops:
                print('%-4s %21.15fd0 %21.15fd0 %21.15fd0' % (species, pos[0], pos[1], pos[2]), file=finp)
        else:
            for i, (species, pos) in enumerate(self.specprops):
                print('%-4s %21.15fd0 %21.15fd0 %21.15fd0   %d  %d  %d' % (species,pos[0],pos[1],pos[2],simpleconstr[i][0],simpleconstr[i][1],simpleconstr[i][2]), file=finp)

        if len(otherconstr) != 0:
            print('CONSTRAINTS', file=finp)
            if self.constr_tol is None:
                print(len(otherconstr), file=finp)
            else:
                print(len(otherconstr), num2str(self.constr_tol), file=finp)
            for x in otherconstr:
                print(x, file=finp)

        if overridekpts is None:
            kp = self.kpts
        else:
            kp = overridekpts
        if isinstance(kp, str):
            if kp == 'gamma':
                print('K_POINTS Gamma', file=finp)
        else:
            if kp.ndim == 1:
                print('K_POINTS automatic', file=finp)
                print('{0:d} {1:d} {2:d} '.format(
                            *kp), file=finp, end=' ')
                if overridekptshift is None:
                    print('{0:d} {1:d} {2:d} '.format(
                            *self.kptshift), file=finp)
                else:
                    print('{0:d} {1:d} {2:d} '.format(
                            *overridekptshift), file=finp)
            else:
                print('K_POINTS crystal', file=finp)
                nrows, ncols = kp.shape
                print(newstr(nrows), file=finp)
                w = 1.0 / nrows
                for row in kp:
                    if ncols == 3:
                        print('{0:24.15e} {1:24.15e} {2:24.15e} {w:24.15e}'.format(
                            *row, w=w), file=finp)
                    else:
                        print('{0:24.15e} {1:24.15e} {2:24.15e} {3:24.15e}'.format(
                            *row), file=finp)

        # closing PWscf input file
        finp.close()
        if self.verbose == 'high':
            print('\nPWscf input file {} written\n'.format(fname))

    def read_forces(self, getall=False):
        '''
        Read the forces from the PWSCF output file

        Args:
            getall (bool) :
                If ``True`` the forces for all relaxation steps are returned,
                in other case only the last configuration is returned.

        Returns:
            forceslist (`array_like` or `list` of `array_like`) :
                Forces
        '''

        forceslist = []

        with open(self.log, 'rU') as fout:
            lines = fout.readlines()

        forcestart = re.compile(r'     Forces acting on atoms \((?:cartesian axes, )?Ry/au\):')
        forceend = '     Total force = '

        startlnos = [no for no, line in enumerate(lines) if forcestart.match(line)]
        endlnos = [no for no, line in enumerate(lines) if forceend in line]

        forcelinenos = zip(startlnos, endlnos)

        for start, end in forcelinenos:
            forces = np.zeros((self.natoms, 3), dtype=float)
            for line in lines[start: end]:
                linesplit = line.split()
                if len(linesplit) > 0 and linesplit[0] == 'atom':
                    fxyz = [float(x) for x in linesplit[-3:]]
                    atom_number = int(linesplit[1]) - 1
                    forces[atom_number] = fxyz
            forces *= Rydberg / Bohr
            forceslist.append(forces)

        del lines

        if len(forceslist) > 0:
            if getall:
                return forceslist
            else:
                return forceslist[-1]
        else:
            raise ValueError('No forces found in pw log')

    def read_energies(self, getall=False):
        '''
        Read the energies from the PWSCF output file

        The method read total energies (zero point) and smearing contributions
        (-TS) and returns either a tuple ``(energy, free_energy)`` or a list
        of such tuples

        Args:
            getall (bool) :
                If ``True`` the energies for all relaxation steps are returned,
                in other case only the last configuration is returned.

        Returns:
            energylist (`tuple` or `list` of `tuple`) :
                Energies
        '''

        with open(self.log, 'r', newline=None) as fout:
            lines = fout.readlines()

        energylinenos = [no for no, line in enumerate(lines) if
                         ('!' in line and 'total energy' in line)]

        tslinenos = [no for no, line in enumerate(lines) if
                     ('     smearing contrib. (-TS)' in line)]

        free_energies = [float(lines[no].split()[-2]) * Rydberg
                         for no in energylinenos]
        ts = [float(lines[no].split()[-2]) * Rydberg for no in tslinenos]
        energies = [fe - sm for fe, sm in zip(free_energies, ts)]
        del lines

        energylist = list(zip(energies, free_energies))

        if getall:
            return energylist
        else:
            return energylist[-1]

    def read_cell(self, getall=False):
        '''
        Read unit cell parameters from the PWSCF output file

        Args:
            getall (bool) :
                If ``True`` the cells for all relaxation steps are returned,
                in other case only the last cell is returned.

        Returns:
            celllist (`array_like` or `list` of `array_like`) :
                Either a 3x3 array with unit cell parameters or list of such
                arrays
        '''

        celllist = []

        with open(self.log, 'rU') as fout:
            lines = fout.readlines()

        # additional unit cell information
        bli_lines = [line for line in lines if 'bravais-lattice index' in line]
        brav_latt_indices = [int(line.split('=')[1].strip()) for line in bli_lines]

        lp_lines = [line for line in lines if 'lattice parameter (alat)' in line]
        lattice_parameters = [float(line.strip().split('=')[1].strip().split()[0])*Bohr for line in lp_lines]

        ca_linenos = [no for no, line in enumerate(lines) if
                      'crystal axes: (cart. coord. in units of alat)' in line]

        for i, no in enumerate(ca_linenos):
            cell = np.zeros((3, 3), dtype=float)

            for number, line in enumerate(lines[no + 1: no + 4]):
                line = line.split('=')[1].strip()[1:-1]
                values = [float(value) for value in line.split()]
                cell[number, :] = values
                cell *= lattice_parameters[i]
            celllist.append(cell)

        del lines

        if getall:
            return celllist
        else:
            return celllist[-1]

    def read_positions(self, getall=False):
        '''
        Read ion positions from the PWSCF output file

        Args:
            getall (bool) :
                If ``True`` the positions for all relaxation steps are
                returned, in other case only the last consiguration is
                returned.

        Returns:
            positionslist (`tuple` or `list` of `tuple`) :
                A tuple contains list of symbols and an array of ion positions
        '''

        positionslist = []

        with open(self.log, 'rU') as fout:
            lines = fout.readlines()

        carteslinenos = [no for no, line in enumerate(lines)
                         if 'Cartesian axes' in line]
        crystallinenos = [no for no, line in enumerate(lines)
                          if 'ATOMIC_POSITIONS (crystal)' in line]

        for no in carteslinenos:
            positions = np.zeros((self.natoms, 3), dtype=float)
            symbols = []
            for line in lines[no + 3:]:
                words = line.split()
                if len(words) == 0:
                    break
                atom_number = int(words[0]) - 1
                xyz = [float(x) for x in words[-4:-1]]
                positions[atom_number] = xyz
                symbols.append(words[1].strip('0123456789'))

            positionslist.append((symbols, positions))

        del lines

        if getall:
            return positionslist
        else:
            return positionslist[-1]

    def read_stress(self, getall=False):
        '''
        Read the stress from the PWSCF output file

        Args:
            getall (bool) :
                If ``True`` the forces for all relaxation steps are returned,
                in other case only the last configuration is returned.

        Returns:
            stresslist (`array_like` or `list` of `array_like`) :
                Stress
        '''

        stresslist = []

        with open(self.log, 'rU') as fout:
            lines = fout.readlines()

        stressstr = '          total   stress  (Ry/bohr**3) '
        stresslnos = [no for no, line in enumerate(lines) if stressstr in line]

        if len(stresslnos) == 0:
            return None

        for lineno in stresslnos:
            stress = np.zeros((3, 3), dtype=float)
            for nrow, line in enumerate(lines[lineno + 1: lineno + 4]):
                linesplit = line.split()
                stressrow = [float(x) for x in linesplit[:3]]
                stress[nrow] = stressrow
            # ASE convention for the stress tensor appears to differ
            # from the PWscf one by a factor of -1
            stress = -1.0 * stress * Rydberg / Bohr**3
            stresslist.append(stress)

        del lines

        if getall:
            return stresslist
        else:
            return stresslist[-1]

    def read_magnetizations(self, getall=False):
        'Read the total and absolute magnetizations from the pw.x log file'

        start = '!    total energy'
        end = '     convergence has'

        mags = dict()
        mags['total magnetization'] = []
        mags['absolute magnetization'] = []

        with open(self.log, 'rU') as fobj:
            for line in fobj:
                if start in line:
                    while end not in line:
                        line = next(fobj)
                        for mag in mags.keys():
                            if mag in line:
                                mags[mag].append(float(line.split()[3]))
        if getall:
            return mags
        else:
            return [mags[k][-1] if mags[k] else 0.0 for k in
                    sorted(mags.keys())]

    def topath(self, filename):
        if os.path.isabs(filename):
            return filename
        else:
            return os.path.join(self.site.submitdir, filename)

    def save_output(self, filename='calc.tgz'):
        """
        Save the contents of calc.save directory.
        """
        file = self.topath(filename)
        self.update(self.atoms)
        self.stop()

        os.system('tar czf ' + filename + ' --directory=' + self.scratch + ' calc.save')

    def load_output(self, filename='calc.tgz'):
        """
        Restore the contents of previously saved calc.save directory.
        """
        self.stop()
        file = self.topath(filename)

        os.system('tar xzf ' + filename + ' --directory=' + self.scratch)

    def save_flev_output(self, filename='calc.tgz'):
        """
        Save the contents of calc.save directory + Fermi level
        & on-site density matrices (if present).
        """
        file = self.topath(filename)
        self.update(self.atoms)

        ef = self.get_fermi_level()
        with open(self.scratch + '/calc.save/fermilevel.txt', 'w') as ftxt:
            ftxt.write('{0:.15e}\n#Fermi level in eV'.format(ef))

        os.system('tar czf '+filename+' --directory='+self.scratch+' calc.save `find . -name "calc.occup*";find . -name "calc.paw"`')

    def load_flev_output(self, filename='calc.tgz'):
        """
        Restore the contents of previously saved calc.save directory
        + Fermi level & on-site density matrices (if present).
        """
        self.stop()
        file = self.topath(filename)

        os.system('tar xzf ' + filename + ' --directory=' + self.scratch)

        self.fermi_input = True
        with open(self.scratch + '/calc.save/fermilevel.txt', 'r') as ftxt:
            self.inputfermilevel = float(ftxt.readline())

    def save_chg(self, filename='chg.tgz'):
        """
        Save charge density.
        """

        self.update(self.atoms)
        self.stop()

        patterns = ["spin-polarization.*", "magnetization.*", "calc.occup*",
                    "calc.paw"]
        files = [str(x) for patt in patterns for x in self.scratch.files(patt)]

        fpath = self.topath(filename)
        with tarfile.open(fpath, mode='w:gz') as tgz:
            for f in files:
                tgz.add(f)

    def load_chg(self, filename='chg.tgz'):
        """
        Extract charge density.
        """

        self.stop()
        fpath = self.topath(filename)

        if os.path.exists(fpath):
            with tarfile.open(fpath, mode='r:gz') as tgz:
                tgz.extractall(path=self.scratch)
        else:
            raise OSError('File <{}> does not exist'.format(fpath))

    def save_wf(self, filename='wf.tgz'):
        """
        Save wave functions.

        Excluding the calc.save directory
        """

        self.update(self.atoms)
        self.stop()

        files = [str(x) for x in self.scratch.files()]

        fpath = self.topath(filename)
        with tarfile.open(fpath, mode='w:gz') as tgz:
            for f in files:
                tgz.add(f)

    def load_wf(self, filename='wf.tgz'):
        """
        Extract the wave functions.
        """

        self.stop()

        fpath = self.topath(filename)
        if os.path.exists(fpath):
            with tarfile.open(fpath, mode='r:gz') as tgz:
                tgz.extractall(path=self.scratch)
        else:
            raise OSError('File <{}> does not exist'.format(fpath))

    def save_flev_chg(self, filename='chg.tgz'):
        """
        Save charge density and Fermi level.
        Useful for subsequent bandstructure or density of states
        calculations.
        """
        file = self.topath(filename)
        self.update(self.atoms)

        ef = self.get_fermi_level()
        with open(self.scratch + '/calc.save/fermilevel.txt', 'w') as ftxt:
            ftxt.write('{0:.15e}\n#Fermi level in eV'.format(ef))

        os.system('tar czf '+filename+' --directory='+self.scratch+' calc.save/charge-density.dat calc.save/data-file.xml `cd '+self.scratch+';find calc.save -name "spin-polarization.*";find calc.save -name "magnetization.*";find . -name "calc.occup*";find . -name "calc.paw"` calc.save/fermilevel.txt')

    def load_flev_chg(self, filename='efchg.tgz'):
        """
        Load charge density and Fermi level.
        Useful for subsequent bandstructure or density of states
        calculations.
        """
        self.stop()
        file = self.topath(filename)

        os.system('tar xzf ' + filename + ' --directory=' + self.scratch)
        self.fermi_input = True
        with open(self.scratch + '/calc.save/fermilevel.txt', 'r') as ftxt:
            self.inputfermilevel = float(ftxt.readline())

    def get_final_structure(self):
        """
        returns Atoms object according to a structure
        optimized internally by quantum espresso
        """
        from ase import Atoms

        self.stop()

        p = os.popen('grep -n Giannozzi ' + self.log + '| tail -1', 'r')
        n = int(p.readline().split()[0].strip(':'))
        p.close()

        s = open(self.log,'r')
        #skip over previous runs in log in case the current log has been
        #appended to old ones
        for i in range(n):
            s.readline()

        a = s.readline()
        while a[:11] != '     celldm':
            a = s.readline()
        alat = float(a.split()[1])/1.889726
        a = s.readline()
        while a[:12] != '     crystal':
            a = s.readline()
        cell = []
        for i in range(3):
            cell.append([float(x) for x in s.readline().split()[3:6]])
        cell = np.array(cell)
        a = s.readline()
        while a[:12] != '     site n.':
            a = s.readline()
        pos = []
        syms = ''
        y = s.readline().split()
        while len(y) > 0:
            nf = len(y)
            pos.append([float(x) for x in y[nf-4:nf-1]])
            syms += y[1].strip('0123456789')
            y = s.readline().split()
        pos = np.array(pos)*alat
        natoms = len(pos)

        #create atoms object with coordinates and unit cell
        #as specified in the initial ionic step in log
        atoms = Atoms(syms, pos, cell=cell*alat, pbc=(1, 1, 1))

        coord = 'angstrom)'
        a = s.readline()
        while a != '':
            while a[:7] != 'CELL_PA' and a[:7] != 'ATOMIC_' and a != '':
                a = s.readline()
            if a == '':
                break
            if a[0] == 'A':
                coord = a.split('(')[-1]
                for i in range(natoms):
                    pos[i][:] = s.readline().split()[1:4]
            else:
                for i in range(3):
                    cell[i][:] = s.readline().split()
            a = s.readline()

        atoms.set_cell(cell*alat, scale_atoms=False)

        if coord == 'alat)':
            atoms.set_positions(pos*alat)
        elif coord == 'bohr)':
            atoms.set_positions(pos*Bohr)
        elif coord == 'angstrom)':
            atoms.set_positions(pos)
        else:
            atoms.set_scaled_positions(pos)

        return atoms

    def get_potential_energy(self, atoms=None, 
                             force_consistent=False
                             #force_consistent=True
                             ):
        self.update(atoms)
            
        if force_consistent or self.force_consistent:
            return self.energy_free
        else:
            return self.energy_zero

    def get_nonselfconsistent_energies(self, type='beefvdw'):
        #assert self.xc is 'BEEF'
        self.stop()
        p = os.popen('grep -32 "BEEF-vdW xc energy contributions" '+self.log+' | tail -32','r')
        s = p.readlines()
        p.close()
        xc = np.array([])
        for i, l in enumerate(s):
            l_ = float(l.split(":")[-1]) * Rydberg
            xc = np.append(xc, l_)
        assert len(xc) == 32
        return xc

    def get_xc_functional(self):
        return self.xc

    def get_stress(self, atoms=None):

        self.update(atoms)
        return self.results['stress']

    def get_absolute_magnetization(self, atoms=None):

        self.update(atoms)
        return self.results.get('absolute magnetization', None)

    def get_total_magnetization(self, atoms=None):

        self.update(atoms)
        return self.results.get('total magnetization', None)

    def get_smearing_contribution(self):
        return self.ST

    def checkerror(self):
        '''
        Check if any errors occured during the calculation
        '''

        with open(self.log, 'r') as flog:
            loglines = flog.readlines()

        gnos = [no for no, line in enumerate(loglines) if 'Giannozzi' in line]
        if len(gnos) < 1:
            raise RuntimeError("Espresso executable doesn't seem to have been started.")

        lnos = [no for no, line in enumerate(loglines) if '%' * 78 in line]

        if len(lnos) < 2:
            return
        else:
            msg = ' '.join(loglines[lnos[-2] + 1: lnos[-1]])
            if msg.lower().find('error') < 0:
                return
            else:
                raise RuntimeError(msg.rstrip('\n'))

    def check_spinpol(self):
        mm = self.atoms.get_initial_magnetic_moments()
        sp = mm.any()
        self.summed_magmoms = np.sum(mm)
        if sp:
            if not self.spinpol and not self.noncollinear:
                raise KeyError('Explicitly specify spinpol=True or noncollinear=True for spin-polarized systems')
            elif abs(self.sigma) <= self.sigma_small and not self.fix_magmom:
                raise KeyError('Please use fix_magmom=True for sigma=0.0 eV and spinpol=True. Hopefully this is not an extended system...?')
        else:
            if self.spinpol and abs(self.sigma) <= self.sigma_small:
                self.fix_magmom = True
        if abs(self.sigma) <= self.sigma_small:
            self.occupations = 'fixed'

    def relax_cell_and_atoms(self,
            cell_dynamics='bfgs', # {'none', 'sd', 'damp-pr', 'damp-w', 'bfgs'}
            ion_dynamics='bfgs', # {'bfgs', 'damp'}
            cell_factor=1.2,
            cell_dofree=None,
            fmax=None,
            press=None,
            dpress=None
            ):
        """
        Simultaneously relax unit cell and atoms using Espresso's internal
        relaxation routines.
        fmax,press are convergence limits and dpress is the convergence
        criterion wrt. reaching the target pressure press
        atoms.get_potential_energy() will yield the final energy,
        but to obtain the structure use
        relaxed_atoms = calc.get_final_structure()
        If you want to continue calculations in relax_atoms, use
        relaxed_atoms.set_calculator(some_espresso_calculator)
        """
        self.stop()
        oldcalculation = self.calculation
        oldalgo = self.ion_dynamics
        oldcell = self.cell_dynamics
        oldfactor = self.cell_factor
        oldfree = self.cell_dofree
        self.cell_dynamics = cell_dynamics
        self.ion_dynamics = ion_dynamics
        self.cell_factor = cell_factor
        self.cell_dofree = cell_dofree
        oldfmax = self.fmax
        oldpress = self.press
        olddpress = self.dpress

        if fmax is not None:
            self.fmax = fmax
        if press is not None:
            self.press = press
        if dpress is not None:
            self.dpress = dpress
        self.calculation = 'vc-relax'
        self.recalculate = True
        self.read(self.atoms)
        self.calculation = oldcalculation
        self.ion_dynamics = oldalgo
        self.cell_dynamics = oldcell
        self.cell_factor = oldfactor
        self.cell_dofree = oldfree
        self.fmax = oldfmax
        self.press = oldpress
        self.dpress = olddpress

    def relax_atoms(self,
            ion_dynamics='bfgs', # {'bfgs', 'damp'}
            fmax=None
            ):
        """
        Relax atoms using Espresso's internal relaxation routines.
        fmax is the force convergence limit
        atoms.get_potential_energy() will yield the final energy,
        but to obtain the structure use
        relaxed_atoms = calc.get_final_structure()
        If you want to continue calculations in relax_atoms, use
        relaxed_atoms.set_calculator(some_espresso_calculator)
        """
        self.stop()
        oldcalculation = self.calculation
        oldalgo = self.ion_dynamics
        self.ion_dynamics=ion_dynamics
        oldfmax = self.fmax

        self.calculation='relax'
        if fmax is not None:
            self.fmax = fmax
        self.recalculate=True
        self.read(self.atoms)
        self.calculation = oldcalculation
        self.ion_dynamics = oldalgo
        self.fmax = oldfmax

    def run_espressox(self, binary, inp, log=None, piperead=False,
                      parallel=True):
        '''
        runs one of the .x binaries of the espresso suite
        inp is expected to be in self.localtmp
        log will be created in self.localtmp
        '''

        if log is None:
            ll = ''
        else:
            ll = ' >>'+self.localtmp+'/'+log
        if self.site.batchmode and parallel:
            cdir = os.getcwd()
            os.chdir(self.localtmp)
            os.system(self.site.perHostMpiExec+' cp '+self.localtmp+'/'+inp+' '+self.scratch)
            if piperead:
                p = self.site.do_perProcMpiExec_outputonly(self.scratch, binary+' '+self.parflags+' -in '+inp+ll)
            else:
                self.site.runonly_perProcMpiExec(self.scratch, binary+' '+self.parflags+' -in '+inp+ll)
            os.chdir(cdir)
        else:
            os.system('cp '+self.localtmp+'/'+inp+' '+self.scratch)
            if piperead:
                p = os.popen('cd '+self.scratch+' ; '+binary+' -in '+inp+ll)
            else:
                os.system('cd '+self.scratch+' ; '+binary+' -in '+inp+ll)
        if piperead:
            return p

    def run_ppx(self, inp, log=None, inputpp=[], plot=[],
        output_format=5, iflag=3, piperead=False, parallel=True):
        if 'disk_io' in list(self.output.keys()):
            if self.output['disk_io'] == 'none':
                print("run_ppx requires output['disk_io'] to be at least 'low' and avoidio=False")
        self.stop()
        finp = open(self.localtmp+'/'+inp, 'w')
        print('&INPUTPP\n  prefix=\'calc\',\n  outdir=\'.\',', file=finp)
        for a,b in inputpp:
            if type(b)==float:
                c = num2str(b)
            elif type(b)==str:
                c = "'"+b+"'"
            else:
                c = str(b)
            print('  '+a+'='+c+',', file=finp)
        print('/', file=finp)
        print('&PLOT\n  iflag=%d,\n  output_format=%d,' % (iflag, output_format), file=finp)
        for a,b in plot:
            if type(b)==float:
                c = num2str(b)
            elif type(b)==str:
                c = "'"+b+"'"
            else:
                c = str(b)
            print('  '+a+'='+c+',', file=finp)
        print('/', file=finp)
        finp.close()

        if piperead:
            return self.run_espressox('pp.x', inp, log=log,
                piperead=piperead, parallel=parallel)
        else:
            self.run_espressox('pp.x', inp, log=log, parallel=parallel)


    def get_fermi_level(self):
        if self.fermi_input:
            return self.inputfermilevel
        self.stop()
        try:
            p = os.popen('grep Fermi '+self.log+'|tail -1', 'r')
            efermi = float(p.readline().split()[-2])
            p.close()
        except:
            raise RuntimeError('get_fermi_level called before DFT calculation was run')
        return efermi

    def calc_pdos(self,
        Emin = None,
        Emax = None,
        DeltaE = None,
        nscf = False,
        tetrahedra = False,
        slab = False,
        kpts = None,
        kptshift = None,
        nbands = None,
        ngauss = None,
        sigma = None,
        nscf_fermilevel=False,
        add_higher_channels=False,
        get_overlap_integrals=False):
        """
        Calculate (projected) density of states.

        - Emin,Emax,DeltaE define the energy window.
        - nscf=True will cause a non-selfconsistent calculation to be performed
          on top of a previous converged scf calculation, with the advantage
          that more kpts and more nbands can be defined improving the quality/
          increasing the energy range of the DOS.
        - tetrahedra=True (in addition to nscf=True) means use tetrahedron
          (i.e. smearing-free) method for DOS
        - slab=True: use triangle method insead of tetrahedron method
          (for 2D system perp. to z-direction)
        - sigma != None sets/overrides the smearing to calculate the DOS
          (also overrides tetrahedron/triangle settings)
        - get_overlap_integrals=True: also return k-point- and band-resolved
          projections (which are summed up and smeared to obtain the PDOS)

        Returns an array containing the energy window,
        the DOS over the same range,
        and the PDOS as an array (index: atom number 0..n-1) of dictionaries.
        The dictionary keys are the angular momentum channels 's','p','d'...
        (or e.g. 'p,j=0.5', 'p,j=1.5' in the case of LS-coupling).
        Each dictionary contains an array of arrays of the total and
        m-resolved PDOS over the energy window.
        In case of spin-polarization, total up is followed by total down, by
        first m with spin up, etc...

        Quantum Espresso with the tetrahedron method for PDOS can be obtained here:

        .. code-block:: bash

           svn co --username anonymous http://qeforge.qe-forge.org/svn/q-e/branches/espresso-dynpy-beef

        """

        efermi = self.get_fermi_level()

        # run a nscf calculation with e.g. tetrahedra or more k-points etc.
        if nscf:
            if not hasattr(self, 'natoms'):
                self.atoms2species()
                self.natoms = len(self.atoms)
            self.write_input(filename='pwnscf.inp',
                calculation='nscf', usetetrahedra=tetrahedra, overridekpts=kpts,
                overridekptshift=kptshift, overridenbands=nbands,
                suppressforcecalc=True)
            self.run_espressox('pw.x', 'pwnscf.inp', 'pwnscf.log')
            if nscf_fermilevel:
                p = os.popen('grep Fermi '+self.localtmp+'/pwnscf.log|tail -1', 'r')
                efermi = float(p.readline().split()[-2])
                p.close()

        # remove old wave function projections
        os.system('rm -f '+self.scratch+'/*_wfc*')
        # create input for projwfc.x
        fpdos = open(self.localtmp+'/pdos.inp', 'w')
        print('&PROJWFC\n  prefix=\'calc\',\n  outdir=\'.\',', file=fpdos)
        if Emin is not None:
            print('  Emin = '+num2str(Emin+efermi)+',', file=fpdos)
        if Emax is not None:
            print('  Emax = '+num2str(Emax+efermi)+',', file=fpdos)
        if DeltaE is not None:
            print('  DeltaE = '+num2str(DeltaE)+',', file=fpdos)
        if slab:
            print('  lslab = .true.,', file=fpdos)
        if ngauss is not None:
            print('  ngauss = '+str(ngauss)+',', file=fpdos)
        if sigma is not None:
            print('  degauss = '+num2str(sigma/Rydberg)+',', file=fpdos)
        print('/', file=fpdos)
        fpdos.close()
        # run projwfc.x
        self.run_espressox('projwfc.x', 'pdos.inp', 'pdos.log')

        # read in total density of states
        dos = np.loadtxt(self.scratch+'/calc.pdos_tot')
        if len(dos[0]) > 3:
            nspin = 2
            self.dos_total = [dos[:,1],dos[:,2]]
        else:
            nspin = 1
            self.dos_total = dos[:,1]
        self.dos_energies = dos[:,0] - efermi
        npoints = len(self.dos_energies)

        channels = {'s':0, 'p':1, 'd':2, 'f':3}
        # read in projections onto atomic orbitals
        self.pdos = [{} for i in range(self.natoms)]
        p = os.popen('ls '+self.scratch+'/calc.pdos_atm*')
        proj = p.readlines()
        p.close()
        proj.sort()
        for i,inp in enumerate(proj):
            inpfile = inp.strip()
            pdosinp = np.genfromtxt(inpfile)
            spl = inpfile.split('#')
            iatom = int(spl[1].split('(')[0])-1
            channel = spl[2].split('(')[1].rstrip(')').replace('_j',',j=')
            jpos = channel.find('j=')
            if jpos<0:
                #ncomponents = 2*l+1 +1  (latter for m summed up)
                ncomponents = (2*channels[channel[0]]+2) * nspin
            else:
                #ncomponents = 2*j+1 +1  (latter for m summed up)
                ncomponents = int(2.*float(channel[jpos+2:]))+2
            if channel not in list(self.pdos[iatom].keys()):
                self.pdos[iatom][channel] = np.zeros((ncomponents,npoints), np.float)
                first = True
            else:
                first = False
            if add_higher_channels or first:
                for j in range(ncomponents):
                    self.pdos[iatom][channel][j] += pdosinp[:,(j+1)]

        if get_overlap_integrals:
            return self.dos_energies, self.dos_total, self.pdos, self.__get_atomic_projections__()
        else:
            return self.dos_energies, self.dos_total, self.pdos

    def calc_bandstructure(self,
        kptpath,
        nbands = None,
        atomic_projections = False):
        """
        Calculate bandstructure along kptpath (= array of k-points).
        If nbands is not None, override number of bands set in calculator.
        If atomic_projections is True, calculate orbital character of
        each band at each k-point.

        Returns an array of energies.
        (if spin-polarized spin is first index;
        the next index enumerates the k-points)
        """

        efermi = self.get_fermi_level()

        # run a nscf calculation
        if not hasattr(self, 'natoms'):
            self.atoms2species()
            self.natoms = len(self.atoms)
        oldnoinv = self.noinv
        oldnosym = self.nosym
        self.noinv = True
        self.nosym = True
        self.write_input(filename='pwnscf.inp',
            calculation='nscf', overridekpts=kptpath,
            overridenbands=nbands, suppressforcecalc=True)
        self.noinv = oldnoinv
        self.nosym = oldnosym
        self.run_espressox('pw.x', 'pwnscf.inp', 'pwnscf.log')

        energies = self.get_eigenvalues(efermi=efermi)

        if not atomic_projections:
            return energies
        else:
            #run pdos calculation with (tiny) E-range
            #to trigger calculation of atomic_proj.xml

            # create input for projwfc.x
            fpdos = open(self.localtmp+'/pdos.inp', 'w')
            print('&PROJWFC\n  prefix=\'calc\',\n  outdir=\'.\',', file=fpdos)
            print('  filpdos = \'projtmp\',', file=fpdos)
            print('  Emin = '+num2str(-0.3+efermi)+',', file=fpdos)
            print('  Emax = '+num2str(-0.2+efermi)+',', file=fpdos)
            print('  DeltaE = 0.1d0,', file=fpdos)
            print('/', file=fpdos)
            fpdos.close()
            # run projwfc.x
            self.run_espressox('projwfc.x', 'pdos.inp', 'pdos.log')
            #remove unneeded pdos files containing only a tiny E-range of two points
            os.system('rm -f ' + self.scratch + '/projtmp*')

            return energies, self.__get_atomic_projections__()


    def __get_atomic_projections__(self):
        f = open(self.scratch+'/calc.save/atomic_proj.xml', 'r')
        p = os.popen('grep -n Giannozzi '+self.localtmp+'/pdos.log|tail -1', 'r')
        n = p.readline().split()[0].strip(':').strip()
        p.close()
        p = os.popen('tail -n +'+n+' '+self.localtmp+'/pdos.log|grep "state #"', 'r')
        #identify states from projwfc.x's stdout
        states = []
        for x in p.readlines():
            y = x.split('atom')[1]
            iatom = int(y.split()[0])-1
            z = y.replace(')\n','').split('=')
            if y.find('m_j')<0:
                l = int(z[1].replace('m',''))
                m = int(z[2])
                states.append([iatom,l,m])
            else:
                j = float(z[1].replace('l',''))
                l = int(z[2].replace('m_j',''))
                mj = float(z[3])
                states.append([iatom,j,l,mj])
        p.close()

        #read in projections from atomic_proj.xml
        a = f.readline()
        while a.find('<NUMBER_OF_B') < 0:
            a = f.readline()
        nbnd = int(f.readline().strip())
        a = f.readline()
        while a.find('<NUMBER_OF_K') < 0:
            a = f.readline()
        nkp = int(f.readline().strip())
        a = f.readline()
        while a.find('<NUMBER_OF_S') < 0:
            a = f.readline()
        spinpol = int(f.readline().strip()) == 2

        if spinpol:
            proj1 = []
            proj2 = []
            proj = proj1
        else:
            proj = []

        while a.find('<ATM') < 0 and a != '':
            a = f.readline()
        if a == '':
            raise RuntimeError('no projections found')

        while True:
            while a.find('<ATM') < 0 and a != '':
                if spinpol and a.find('<SP') >= 0:
                    if a.find('N.1') > 0:
                        proj = proj1
                    else:
                        proj = proj2
                a = f.readline()
            if a == '':
                break
            pr = np.empty(nbnd, np.complex)
            for i in range(nbnd):
                b = f.readline().split(',')
                pr[i] = float(b[0]) + 1j*float(b[1])
            proj.append(pr)
            a = f.readline()

        f.close()

        if spinpol:
            projections = np.array([proj1, proj2])
            return states, np.reshape(projections, (2, nkp, len(proj1)/nkp, nbnd))
        else:
            projections = np.array(proj)
            return states, np.reshape(projections, (nkp, len(proj)/nkp, nbnd))

    def get_eigenvalues(self, kpt=None, spin=None, efermi=None):
        self.stop()

        if self.spinpol:
            p = os.popen("grep eigenval1.xml "+self.scratch+"/calc.save/data-file.xml|tr '\"' ' '|awk '{print $(NF-1)}'", 'r')
            kptdirs1 = [x.strip() for x in p.readlines()]
            p.close()
            kptdirs1.sort()
            p = os.popen("grep eigenval2.xml "+self.scratch+"/calc.save/data-file.xml|tr '\"' ' '|awk '{print $(NF-1)}'", 'r')
            kptdirs2 = [x.strip() for x in p.readlines()]
            p.close()
            kptdirs2.sort()
            kptdirs = kptdirs1+kptdirs2
        else:
            p = os.popen("grep eigenval.xml "+self.scratch+"/calc.save/data-file.xml|tr '\"' ' '|awk '{print $(NF-1)}'", 'r')
            kptdirs = [x.strip() for x in p.readlines()]
            p.close()
            kptdirs.sort()

        nkp2 = len(kptdirs)/2
        if kpt is None: #get eigenvalues at all k-points
            if self.spinpol:
                if spin == 'up' or spin == 0:
                    kp = kptdirs[:nkp2]
                if spin == 'down' or spin == 1:
                    kp = kptdirs[nkp2:]
                else:
                    kp = kptdirs
            else:
                kp = kptdirs
        else:           #get eigenvalues at specific k-point
            if self.spinpol:
                if spin == 'up' or spin == 0:
                    kp = [kptdirs[kpt]]
                if spin == 'down' or spin == 1:
                    kp = [kptdirs[kpt+nkp2]]
                else:
                    kp = [kptdirs[kpt], kptdirs[kpt + nkp]]
            else:
                kp = [kptdirs[kpt]]

        if efermi is None:
            ef = 0.
        else:
            ef = efermi

        eig = []
        for k in kp:
            f = open(self.scratch + '/calc.save/' + k, 'r')
            a = f.readline()
            while a.upper().find('<EIG') < 0:
                a = f.readline()
            nbnd = int(a.split('"')[-2])
            eig.append(Hartree*np.fromfile(f, dtype=float, count=nbnd, sep=' ') - ef)
            f.close()

        spinall = spin not in ('up','down', 0, 1)
        if kpt is not None and spinall:
            return np.array(eig[0])
        elif kpt is None and spinall and self.spinpol:
            return np.reshape(np.array(eig), (2, nkp2, nbnd))
        else:
            return np.array(eig)

    def read_3d_grid(self, stream, log):
        f = open(self.localtmp+'/'+log, 'a')
        x = stream.readline()
        while x != '' and x[:11] != 'DATAGRID_3D':
            f.write(x)
            x = stream.readline()
        if x == '':
            raise RuntimeError('error reading 3D data grid')
        f.write(x)
        nx, ny, nz = [int(y) for y in stream.readline().split()]
        origin = np.array([float(y) for y in stream.readline().split()])
        cell = np.empty((3,3), np.float)
        for i in range(3):
            cell[i][:] = [float(y) for y in stream.readline().split()]
        data = np.reshape(np.fromfile(stream, count=nx*ny*nz, sep=' '),
            (nx,ny,nz), order='F')

        x = stream.readline()
        while x != '':
            f.write(x)
            x = stream.readline()

        f.close()
        return (origin, cell, data)

    def read_2d_grid(self, stream, log):
        f = open(self.localtmp+'/' + log, 'a')
        x = stream.readline()
        while x != '' and x[:11] != 'DATAGRID_2D':
            f.write(x)
            x = stream.readline()
        if x == '':
            raise RuntimeError('error reading 2D data grid')
        f.write(x)
        nx, ny = [int(y) for y in stream.readline().split()]
        origin = np.array([float(y) for y in stream.readline().split()])
        cell = np.empty((3,3), np.float)
        for i in range(3):
            cell[i][:] = [float(y) for y in stream.readline().split()]
        data = np.reshape(np.fromfile(stream, count=nx*ny, sep=' '),
            (nx,ny), order='F')

        x = stream.readline()
        while x != '':
            f.write(x)
            x = stream.readline()

        f.close()
        return (origin,cell,data)

    def get_spin_component(self, spin='both'):
        'Get the spin component as interger'

        if spin == 'both' or spin == 0:
            s = 0
        elif spin == 'up' or spin == 1:
            s = 1
        elif spin == 'down' or spin == 2:
            s = 2
        else:
            raise ValueError('unknown spin component {}'.format(spin))

    def extract_charge_density(self, spin='both'):
        """
        Obtains the charge density as a numpy array after a DFT calculation.
        Returns (origin,cell,density).
        """

        s = self.get_spin_component(spin=spin)

        p = self.run_ppx('charge.inp',
            inputpp=[['plot_num',0],['spin_component',s]],
            piperead=True, parallel=False)
        origin,cell,data = self.read_3d_grid(p, 'charge.log')
        p.close()
        return (origin,cell,data)

    def xsf_charge_density(self, xsf, spin='both'):
        """
        Writes the charge density from a DFT calculation
        to an input file for xcrysden.
        """

        s = self.get_spin_component(spin=spin)

        self.run_ppx('charge.inp',
            inputpp=[['plot_num',0],['spin_component',s]],
            plot=[['fileout',self.topath(xsf)]],
            parallel=False, log='charge.log')

    def extract_total_potential(self, spin='both'):
        """
        Obtains the total potential as a numpy array after a DFT calculation.
        Returns (origin,cell,potential).
        """

        s = self.get_spin_component(spin=spin)

        p = self.run_ppx('totalpot.inp',
            inputpp=[['plot_num',1],['spin_component',s]],
            piperead=True, parallel=False)
        origin,cell,data = self.read_3d_grid(p, 'totalpot.log')
        p.close()
        return (origin,cell,data*Rydberg)

    def xsf_total_potential(self, xsf, spin='both'):
        """
        Writes the total potential from a DFT calculation
        to an input file for xcrysden.
        """

        s = self.get_spin_component(spin=spin)

        self.run_ppx('totalpot.inp',
            inputpp=[['plot_num',1],['spin_component',s]],
            plot=[['fileout',self.topath(xsf)]],
            parallel=False, log='totalpot.log')

    def extract_local_ionic_potential(self):
        """
        Obtains the local ionic potential as a numpy array after a DFT calculation.
        Returns (origin,cell,potential).
        """
        p = self.run_ppx('vbare.inp',
            inputpp=[['plot_num',2]],
            piperead=True, parallel=False)
        origin,cell,data = self.read_3d_grid(p, 'vbare.log')
        p.close()
        return (origin,cell,data*Rydberg)

    def xsf_local_ionic_potential(self, xsf):
        """
        Writes the local ionic potential from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx('vbare.inp',
            inputpp=[['plot_num',2]],
            plot=[['fileout',self.topath(xsf)]],
            parallel=False, log='vbare.log')

    def extract_local_dos_at_efermi(self):
        """
        Obtains the local DOS at the Fermi level as a numpy array after a DFT calculation.
        Returns (origin,cell,ldos).
        """
        p = self.run_ppx('ldosef.inp',
            inputpp=[['plot_num',3]],
            piperead=True, parallel=False)
        origin,cell,data = self.read_3d_grid(p, 'ldosef.log')
        p.close()
        return (origin,cell,data)

    def xsf_local_dos_at_efermi(self, xsf):
        """
        Writes the local DOS at the Fermi level from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx('ldosef.inp',
            inputpp=[['plot_num',3]],
            plot=[['fileout',self.topath(xsf)]],
            parallel=False, log='ldosef.log')

    def extract_local_entropy_density(self):
        """
        Obtains the local entropy density as a numpy array after a DFT calculation.
        Returns (origin,cell,density).
        """
        p = self.run_ppx('lentr.inp',
            inputpp=[['plot_num',4]],
            piperead=True, parallel=False)
        origin,cell,data = self.read_3d_grid(p, 'lentr.log')
        p.close()
        return (origin,cell,data)

    def xsf_local_entropy_density(self, xsf):
        """
        Writes the local entropy density from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx('lentr.inp',
            inputpp=[['plot_num',4]],
            plot=[['fileout',self.topath(xsf)]],
            parallel=False, log='lentr.log')

    def extract_stm_data(self, bias):
        """
        Obtains STM data as a numpy array after a DFT calculation.
        Returns (origin,cell,stmdata).
        """
        p = self.run_ppx('stm.inp',
            inputpp=[['plot_num',5],['sample_bias',bias/Rydberg]],
            piperead=True, parallel=False)
        origin,cell,data = self.read_3d_grid(p, 'stm.log')
        p.close()
        return (origin,cell,data)

    def xsf_stm_data(self, xsf, bias):
        """
        Writes STM data from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx('stm.inp',
            inputpp=[['plot_num',5],['sample_bias',bias/Rydberg]],
            plot=[['fileout',self.topath(xsf)]],
            parallel=False, log='stm.log')


    def extract_magnetization_density(self):
        """
        Obtains the magnetization density as a numpy array after a DFT calculation.
        Returns (origin,cell,density).
        """
        p = self.run_ppx('magdens.inp',
            inputpp=[['plot_num',6]],
            piperead=True, parallel=False)
        origin,cell,data = self.read_3d_grid(p, 'magdens.log')
        p.close()
        return (origin,cell,data)

    def xsf_magnetization_density(self, xsf):
        """
        Writes the magnetization density from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx('magdens.inp',
            inputpp=[['plot_num',6]],
            plot=[['fileout',self.topath(xsf)]],
            parallel=False, log='magdens.log')

    def extract_wavefunction_density(self, band, kpoint=0, spin='up',
        gamma_with_sign=False):
        """
        Obtains the amplitude of a given wave function as a numpy array after a DFT calculation.
        Returns (origin,cell,amplitude).
        """
        if spin == 'up' or spin == 1:
            s = 0
        elif spin == 'down' or spin == 2:
            s = 1
        elif spin == 'charge' or spin == 0:
            s = 0
        elif spin == 'x':
            s = 1
        elif spin == 'y':
            s = 2
        elif spin == 'z':
            s = 3
        else:
            raise ValueError('unknown spin component')
        if self.spinpol:
            p = os.popen('grep "number of k points=" '+self.log+'|tail -1|tr \'=\' \' \'', 'r')
            nkp = int(p.readline().split()[4])
            p.close()
            kp = kpoint+nkp/2*s
        else:
            kp = kpoint
        inputpp = [['plot_num',7],['kpoint',kp],['kband',band]]
        if gamma_with_sign:
            inputpp.append(['lsign','.true.'])
        if self.noncollinear:
            inputpp.append(['spin_component',s])
        p = self.run_ppx('wfdens.inp',
            inputpp=inputpp,
            piperead=True, parallel=True)
        origin,cell,data = self.read_3d_grid(p, 'wfdens.log')
        p.close()
        return (origin,cell,data)

    def xsf_wavefunction_density(self, xsf, band, kpoint=0, spin='up',
        gamma_with_sign=False):
        """
        Writes the amplitude of a given wave function from a DFT calculation
        to an input file for xcrysden.
        """
        if spin == 'up' or spin == 1:
            s = 0
        elif spin == 'down' or spin == 2:
            s = 1
        elif spin == 'charge' or spin == 0:
            s = 0
        elif spin == 'x':
            s = 1
        elif spin == 'y':
            s = 2
        elif spin == 'z':
            s = 3
        else:
            raise ValueError('unknown spin component')
        if self.spinpol:
            p = os.popen('grep "number of k points=" '+self.log+'|tail -1|tr \'=\' \' \'', 'r')
            nkp = int(p.readline().split()[4])
            p.close()
            kp = kpoint+nkp/2*s
        else:
            kp = kpoint
        inputpp = [['plot_num',7],['kpoint',kp],['kband',band]]
        if gamma_with_sign:
            inputpp.append(['lsign','.true.'])
        if self.noncollinear:
            inputpp.append(['spin_component',s])
        self.run_ppx('wfdens.inp',
            inputpp=inputpp,
            plot=[['fileout',self.topath(xsf)]],
            parallel=True, log='wfdens.log')

    def extract_electron_localization_function(self):
        """
        Obtains the ELF as a numpy array after a DFT calculation.
        Returns (origin,cell,elf).
        """
        p = self.run_ppx('elf.inp',
            inputpp=[['plot_num',8]],
            piperead=True, parallel=False)
        origin,cell,data = self.read_3d_grid(p, 'elf.log')
        p.close()
        return (origin,cell,data)

    def xsf_electron_localization_function(self, xsf):
        """
        Writes the ELF from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx('elf.inp',
            inputpp=[['plot_num',8]],
            plot=[['fileout',self.topath(xsf)]],
            parallel=False, log='elf.log')


    def extract_density_minus_atomic(self):
        """
        Obtains the charge density minus atomic charges as a numpy array after a DFT calculation.
        Returns (origin,cell,density).
        """
        p = self.run_ppx('dens_wo_atm.inp',
            inputpp=[['plot_num',9]],
            piperead=True, parallel=False)
        origin,cell,data = self.read_3d_grid(p, 'dens_wo_atm.log')
        p.close()
        return (origin,cell,data)

    def xsf_density_minus_atomic(self, xsf):
        """
        Writes the charge density minus atomic charges from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx('dens_wo_atm.inp',
            inputpp=[['plot_num',9]],
            plot=[['fileout',self.topath(xsf)]],
            parallel=False, log='dens_wo_atm.log')

    def extract_int_local_dos(self, spin='both', emin=None, emax=None):
        """
        Obtains the integrated ldos as a numpy array after a DFT calculation.
        Returns (origin,cell,ldos).
        """

        s = self.get_spin_component(spin=spin)

        inputpp=[['plot_num',10],['spin_component',s]]
        efermi = self.get_fermi_level()
        if emin is not None:
            inputpp.append(['emin',emin-efermi])
        if emax is not None:
            inputpp.append(['emax',emax-efermi])

        p = self.run_ppx('ildos.inp',
            inputpp=inputpp,
            piperead=True, parallel=False)
        origin,cell,data = self.read_3d_grid(p, 'ildos.log')
        p.close()
        return (origin,cell,data)

    def xsf_int_local_dos(self, xsf, spin='both', emin=None, emax=None):
        """
        Writes the integrated ldos from a DFT calculation
        to an input file for xcrysden.
        """

        s = self.get_spin_component(spin=spin)

        inputpp=[['plot_num',10],['spin_component',s]]
        efermi = self.get_fermi_level()
        if emin is not None:
            inputpp.append(['emin',emin-efermi])
        if emax is not None:
            inputpp.append(['emax',emax-efermi])

        self.run_ppx('ildos.inp',
            inputpp=inputpp,
            plot=[['fileout',self.topath(xsf)]],
            parallel=False, log='ildos.log')


    def extract_ionic_and_hartree_potential(self):
        """
        Obtains the sum of ionic and Hartree potential as a numpy array after a DFT calculation.
        Returns (origin,cell,potential).
        """
        p = self.run_ppx('potih.inp',
            inputpp=[['plot_num',11]],
            piperead=True, parallel=False)
        origin,cell,data = self.read_3d_grid(p, 'potih.log')
        p.close()
        return (origin,cell,data*Rydberg)

    def xsf_ionic_and_hartree_potential(self, xsf):
        """
        Writes the sum of ionic and Hartree potential from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx('potih.inp',
            inputpp=[['plot_num',11]],
            plot=[['fileout',self.topath(xsf)]],
            parallel=False, log='potih.log')


    def extract_sawtooth_potential(self):
        """
        Obtains the saw tooth potential as a numpy array after a DFT calculation.
        Returns (origin,cell,potential).
        """
        p = self.run_ppx('sawtooth.inp',
            inputpp=[['plot_num',12]],
            piperead=True, parallel=False)
        origin,cell,data = self.read_3d_grid(p, 'sawtooth.log')
        p.close()
        return (origin,cell,data*Rydberg)

    def xsf_sawtooth_potential(self, xsf):
        """
        Writes the saw tooth potential from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx('sawtooth.inp',
            inputpp=[['plot_num',12]],
            plot=[['fileout',self.topath(xsf)]],
            parallel=False, log='sawtooth.log')


    def extract_noncollinear_magnetization(self, spin='all'):
        """
        Obtains the non-collinear magnetization as a numpy array after a DFT calculation.
        Returns (origin,cell,magnetization).
        """
        if spin=='all' or spin=='charge' or spin==0:
            s = 0
        elif spin=='x':
            s = 1
        elif spin=='y':
            s = 2
        elif spin=='z':
            s = 3
        else:
            raise ValueError('unknown spin component')
        p = self.run_ppx('noncollmag.inp',
            inputpp=[['plot_num',13],['spin_component',s]],
            piperead=True, parallel=False)
        origin,cell,data = self.read_3d_grid(p, 'noncollmag.log')
        p.close()
        return (origin,cell,data)

    def xsf_noncollinear_magnetization(self, xsf, spin='all'):
        """
        Writes the non-collinear magnetization as from a DFT calculation
        to an input file for xcrysden.
        """
        if spin=='all' or spin=='charge' or spin==0:
            s = 0
        elif spin=='x':
            s = 1
        elif spin=='y':
            s = 2
        elif spin=='z':
            s = 3
        else:
            raise ValueError('unknown spin component')
        self.run_ppx('noncollmag.inp',
            inputpp=[['plot_num',13],['spin_component',s]],
            plot=[['fileout',self.topath(xsf)]],
            parallel=False)

    def extract_ae_charge_density(self, spin='both'):
        """
        Obtains the all-electron (PAW) charge density as a numpy array after a DFT calculation.
        Returns (origin,cell,density)
        """

        s = self.get_spin_component(spin=spin)

        p = self.run_ppx('aecharge.inp',
            inputpp=[['plot_num',17],['spin_component',s]],
            piperead=True, parallel=False)
        origin,cell,data = self.read_3d_grid(p, 'aecharge.log')
        p.close()
        return (origin,cell,data)

    def xsf_ae_charge_density(self, xsf, spin='both'):
        """
        Writes the all-electron (PAW) charge density from a DFT calculation
        to an input file for xcrysden.
        """

        s = self.get_spin_component(spin=spin)

        self.run_ppx('aecharge.inp',
            inputpp=[['plot_num',17],['spin_component',s]],
            plot=[['fileout',self.topath(xsf)]],
            parallel=False, log='aecharge.log')

    def extract_noncollinear_xcmag(self):
        """
        Obtains the xc magnetic field for a non-collinear system as a numpy
        array after a DFT calculation. Returns (origin,cell,field).
        """

        p = self.run_ppx('ncxcmag.inp',
            inputpp=[['plot_num',18]],
            piperead=True, parallel=False)
        origin,cell,data = self.read_3d_grid(p, 'ncxcmag.log')
        p.close()
        return (origin,cell,data)

    def xsf_noncollinear_xcmag(self, xsf):
        """
        Writes the xc magnetic field for a non-collinear system from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx('ncxcmag.inp',
            inputpp=[['plot_num',18]],
            plot=[['fileout',self.topath(xsf)]],
            parallel=False, log='ncxcmag.log')

    def extract_reduced_density_gradient(self):
        """
        Obtains the reduced density gradient as a numpy array after a DFT calculation.
        Returns (origin,cell,gradient).
        """
        p = self.run_ppx('redgrad.inp',
            inputpp=[['plot_num',19]],
            piperead=True, parallel=False)
        origin,cell,data = self.read_3d_grid(p, 'redgrad.log')
        p.close()
        return (origin,cell,data)

    def xsf_reduced_density_gradient(self, xsf):
        """
        Writes the reduced density gradient from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx('redgrad.inp',
            inputpp=[['plot_num',19]],
            plot=[['fileout',self.topath(xsf)]],
            parallel=False, log='redgrad.log')

    def extract_middle_density_hessian_eig(self):
        """
        Obtains the middle Hessian eigenvalue as a numpy array after a DFT calculation.
        Returns (origin,cell,density).
        """
        p = self.run_ppx('mideig.inp',
            inputpp=[['plot_num',20]],
            piperead=True, parallel=False)
        origin,cell,data = self.read_3d_grid(p, 'mideig.log')
        p.close()
        return (origin,cell,data)

    def xsf_middle_density_hessian_eig(self, xsf):
        """
        Writes the middle Hessian eigenvalue from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx('mideig.inp',
            inputpp=[['plot_num',20]],
            plot=[['fileout',self.topath(xsf)]],
            parallel=False, log='mideig.log')


    def find_max_empty_space(self, edir=3):
        """
        Assuming periodic boundary conditions, finds the largest
        continuous segment of free, unoccupied space and returns
        its midpoint in scaled coordinates (0 to 1) in the edir direction (default z).
        """
        position_array = self.atoms.get_scaled_positions()[..., edir - 1]  # 0-indexed direction
        position_array.sort()
        differences = np.diff(position_array)
        differences = np.append(differences, position_array[0] + 1 - position_array[-1])  # through the PBC
        max_diff_index = np.argmax(differences)
        if max_diff_index == len(position_array) - 1:
            return (position_array[0] + 1 + position_array[-1]) / 2. % 1  # should be < 1 in cell units
        else:
            return (position_array[max_diff_index] + position_array[max_diff_index + 1]) / 2.


    def get_work_function(self, pot_filename="pot.xsf", edir=3):
        """
        Calculates the work function of a calculation by subtracting the electrostatic
        potential of the vacuum (from averaging the output of pp.x num_plot 11 in the z
        direction by default) from the Fermi energy.
        Values used for average.x come from the espresso example for work function for a surface
        """
        #TODO: Implement some sort of tuning for these parameters?
        if pot_filename[0] != '/':
            file = os.path.join(self.site.submitdir, pot_filename)
        else:
            file = pot_filename
        self.update(self.atoms)
        self.stop()
        self.run_ppx('wf_pp.in', log='wf_pp.log',
            inputpp=[('plot_num', 11), ('filplot', self.topath('pot.xsf'))],
            output_format=3, iflag=3, piperead=False, parallel=False)

        favg = open(self.localtmp + '/avg.in', 'w')
        print('1', file=favg)
        print(os.path.join(self.site.submitdir, pot_filename), file=favg)
        print('1.D0', file=favg)
        print('1440', file=favg)
        print(str(edir), file=favg)
        print('3.835000000', file=favg)
        print('', file=favg)
        favg.close()
        os.system('cp ' + self.localtmp + '/avg.in ' + self.scratch)
        os.system('cd ' + self.scratch + ' ; ' + 'average.x < avg.in >>' + self.localtmp + '/avg.out')

        # Pick a good place to sample vacuum level
        cell_length = self.atoms.cell[edir - 1][edir - 1] / Bohr
        vacuum_pos = self.find_max_empty_space(edir) * cell_length
        avg_out = open(self.localtmp + '/avg.out', 'r')
        record = False
        average_data = []
        lines = list(avg_out)
        for line in lines:
            if len(line.split()) == 3 and line.split()[0] == "0.000000000":
                record = True
            elif len(line.split()) == 0:
                record = False
            if record == True:
                average_data.append([float(i) for i in line.split()])
        # [1] is planar average [2] is macroscopic average
        vacuum_energy = average_data[np.abs(np.array(average_data)[..., 0] - vacuum_pos).argmin()][1]

        # Get the latest Fermi energy
        fermi_data = os.popen('grep -n "Fermi" ' + self.log + ' | tail -1', 'r')
        fermi_energy = float(fermi_data.readline().split()[-2])
        fermi_data.close()

        # if there's a dipole, we need to return 2 work functions - one for either direction away from the slab
        if self.dipole['status']:
            eopreg = 0.025
            if 'eopreg' in list(self.dipole.keys()):
                eopreg = self.dipole['eopreg']
            # we use cell_length*eopreg*2.5 here since the work functions seem to converge at that distance rather than *1 or *2
            vac_pos1 = (vacuum_pos - cell_length*eopreg*2.5) % cell_length
            vac_pos2 = (vacuum_pos + cell_length*eopreg*2.5) % cell_length
            vac_index1 = np.abs(np.array(average_data)[..., 0] - vac_pos1).argmin()
            vac_index2 = np.abs(np.array(average_data)[..., 0] - vac_pos2).argmin()
            vacuum_energy1 = average_data[vac_index1][1]
            vacuum_energy2 = average_data[vac_index2][1]
            wf = [vacuum_energy1 * Rydberg - fermi_energy, vacuum_energy2 * Rydberg - fermi_energy]
        else:
            wf = vacuum_energy * Rydberg - fermi_energy

        return wf

    def generate_dummy_data(self):
        """
        Generate calc.save/data-file.xml, with non-sense electronic dispersion
        data (1-kpoint and 1 unconverged band), to be able to extract
        charge-density-only-dependent output data in case only the charge-density
        was stored.
        """
        convsave = self.convergence.copy()
        occupationssave = self.occupations
        self.occupations = 'fixed'
        #avoid espresso performing diagonalization
        self.convergence = {'maxsteps': -1,
                            'diag': 'cg',
                            'diago_cg_max_iter': -1,
                            'energy': 1e80}
        if not hasattr(self, 'natoms'):
            self.atoms2species()
            self.natoms = len(self.atoms)
        self.write_input(filename='nonsense.inp',
                         calculation='nscf', overridekpts=(1, 1, 1),
                         overridekptshift=(0, 0, 0), overridenbands=1,
                         suppressforcecalc=True)
        self.run_espressox('pw.x', 'nonsense.inp', 'nonsense.log',
                           parallel=False)
        self.occupations = occupationssave
        del self.convergence
        self.convergence = convsave

    def get_world(self):
        from .worldstub import world
        return world(self.site.nprocs)

    def get_output_header(self):
        '''
        Return a string with information about the execution environment
        '''

        exedir = os.path.dirname(os.popen('which pw.x').readline())

        out = '\n'.join(['# python dir           : {}'.format(self.mypath),
                         '# espresso dir         : {}'.format(exedir),
                         '# pseudo dir           : {}'.format(self.psppath),
                         '# ase-espresso version : {}'.format(self.get_version()),
                         ])
        return out + '\n\n\n'

    def __del__(self):
        try:
            self.stop()
        except:
            pass

    def get_espresso_runtime(self):
        """
        reads the walltime from the espresso output file
        """
        import time
        self.stop()
        time.sleep(0.5)
        p = open(self.log,'r')
        s = p.readlines()[-8]
        p.close()
        #try:
        tmp = s.split('CPU')[1]
        tmp = tmp.split('WALL')[0].strip()
        return tmp
        #except:
        #    return None
        """
        self.stop()
        p = os.popen('grep -a "PWSCF" '+'log'+' | tail -1', 'r')
        s = p.readlines()
        p.close()
        #try:
        tmp = s[0].split('CPU')
        print(tmp)
        return tmp[1].split(' WALL')[0].strip()
        #except:
        #    return None
        """

    def todict(self,only_nondefaults=True,identifier=None):
        """
        Coverts calculator object into a dictionary representation appropriate to be placed
        in a MongoDB. By default this returns only the settings that are not the default values.
        This function is based loosely off the todict function from the Kitchen Group's VASP
        environment. in modifying this function please attempt to conform to the naming 
        conventions found there:
        https://github.com/jkitchin/vasp/blob/master/vasp/vasp_core.py

        only_nondefaults: bool
            If set to True, only the non-default keyword arguments are returned. If False all
            key word arguements are returned
        """
        from collections import OrderedDict
        import getpass
        dict_version = OrderedDict()
        for item in self.defaults.keys():
            if type(getattr(self,item)) == dict:
                dict_version[item] = OrderedDict(getattr(self, item))
            else:
                dict_version[item] = getattr(self, item)
        dict_version['path'] = os.path.abspath('.').split(os.sep)[1:]
        if self.results == {}:
            return dict_version
        #if self.beefensemble == True and self.printensemble == True:
        #    dict_version['beefensemble'] = self.get_beef_ensemble()
        for prop in self.implemented_properties:
            val = self.results.get(prop, None)
            dict_version[prop] = val
        f = self.results.get('forces', None)
        if f is not None:
            dict_version['fmax'] = max(np.abs(f.flatten()))

        s = self.results.get('stress', None)
        if s is not None:
            dict_version['smax'] = max(np.abs(s.flatten()))

        dict_version['nvalence'] = sum(self.get_nvalence()[0])

        dict_version['SCF-steps'] = self.get_number_of_scf_steps()
        dict_version['version'] = self.get_espresso_version()
        try:
            dict_version['psp_path'] = os.environ['ESP_PSP_PATH']
        except:
            pass
        time = self.get_espresso_runtime()

        # convert time from a string to the number of seconds
        if time is not None:
            if 'h' in time:
                hours, minutes = time.split('h')
                time = 3600 * float(hours) + 60 * float(minutes[:-1])
            elif 'm' in time:
                minutes, seconds = time.split('m')
                time = 60 * float(minutes) + float(seconds[:-1])
            elif time is not None:
                time = float(time[:-1])

        if time is not None:
            dict_version['elapsed-time'] = time
        dict_version['name'] = 'espresso'
        if 'parflags' in dict_version:  # We don't care about this
            del dict_version['parflags']
        for item in ['xc','pw','dw','spinpol']: # These should always be in
            dict_version[item] = getattr(self, item)
        """
        for item in dict_version:
            if type(dict_version[item]) not in \
            [dict, list, str, float, int, None, tuple] and \
            dict_version[item] is not None:  # converting numpy arrays to lists
                try:
                    dict_version[item] = dict_version[item].tolist()
                except:
                    pass
        """
        for item in dict_version:
            if type(dict_version[item]) not in \
            [dict, list, str, float, int, None, tuple] and \
            dict_version[item] is not None:  # converting numpy arrays to lists
                try:
                    dict_version[item] = dict_version[item].tolist()
                except:
                    pass
            if item == 'results':
                for key,res in dict_version[item].items():
                    try:
                        dict_version[item][key] = res.tolist()
                    except:
                        pass
        if identifier is not None:
            dict_version['identifier'] = identifier
        
        # Stuff we don't want
        del dict_version['atoms']
        del dict_version['site']
        return dict_version

    def get_espresso_version(self):
        """
        reads the espresso version from the espresso output file
        """
        p = os.popen('grep -a "Program PWSCF" '+self.log+' | tail -1', 'r')
        s = p.readlines()
        p.close()
        tmp = s[0].split('PWSCF ')
        return tmp[1].split(' starts')[0]

    def get_espresso_runtime(self):
        """
        reads the walltime from the espresso output file
        """
        try:
            import time
            time.sleep(0.5)
            p = open(self.log,'r')
            s = p.readlines()[-8]
            p.close()
            #try:
            tmp = s.split('CPU')[1]
            tmp = tmp.split('WALL')[0].strip()
            return tmp
        except:
            return None
        """
        self.stop()
        p = os.popen('grep -a "PWSCF" '+'log'+' | tail -1', 'r')
        s = p.readlines()
        p.close()
        #try:
        tmp = s[0].split('CPU')
        print(tmp)
        return tmp[1].split(' WALL')[0].strip()
        #except:
        #  return None
        """ 

    def calc_to_database(self,
                 host='localhost',
                 port=27017,
                 database='atoms',
                 collection='atoms',
                 user=None,
                 password=None):
        """
        Adds the current calculaton to a mongo database
        """
        from espresso.mongo import mongo_doc, MongoDatabase

        db = MongoDatabase(
                 host=host,
                 port=port,
                 database=database,
                 collection=collection,
                 user=user,
                 password=password)
        atoms = self.atoms  # the atoms object in the calc does not have a calc
        atoms.set_calculator(self)
        id_ = db.write(mongo_doc(atoms))
        return id_




class iEspresso(Espresso):

    '''
    Interactive Quantum Espresso calculator that requires a
    version of the QE binary that supports feeding new coordinates
    after each single point calcualtion.

    Args:
        timeout (int) :
            Timeout for the pexpect.spawn method [in s] that will terminate
            the `expect` the full output in that time, otherwise an
            exception is thrown, defaults to 1800 s
    '''

    def __init__(self, timeout=1800, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.timeout = timeout
        self._spawned = False

    def initialize(self, atoms):
        '''
        Create the scratch directories and pw.inp input file and
        prepare for writing the input file
        '''

        if not self._initialized:
            self.create_outdir()
            self.logfile = open(self.log, 'ab')
            if self.site.usehostfile:
                self.site.write_local_hostfile()

        self.set_pseudo_path()
        self.atoms = atoms.copy()
        self.natoms = len(self.atoms)
        self.atoms2species()

        self.check_spinpol()
        self._initialized = True

    @preserve_cwd
    def run(self):

        if self.site.batchmode:
            self.localtmp.chdir()

            cmd = self.site.get_host_mpi_command('cp -u {0:s} {1:s}'.format(
                str(self.localtmp.joinpath('pw.inp')), self.scratch))
            subprocess.call(cmd)

            if self.calculation != 'hund':

                command = self.site.get_proc_mpi_command(self.scratch,
                                'pw.x ' + self.parflags + ' -in pw.inp', aslist=False)

                if not self._spawned:
                    self.child = pexpect.spawn(command)
                    self._spawned = True
                    self.child.logfile = self.logfile
                    self.child.logfile.write(self.get_output_header().encode('utf-8'))

                    try:
                        i = self.child.expect(['!ASE\s*\n(.*\n){2}',
                            'convergence NOT', 'stopping'],
                                              timeout=self.timeout)
                        if i == 1:
                            raise SCFMaxIterationsError()
                        elif i == 2:
                            raise SCFConvergenceError()
                    except:
                        print('# Exception was thrown by pexpect.expect')
                        print(str(self.child))

                else:  # QE process is already spawned
                    self.child.send('C\n')
                    for atom in self.atoms:
                        self.child.send('{0:25.14e} {1:25.14e} {2:25.10e}\n'.format(atom.x, atom.y, atom.z))
                    try:
                        i = self.child.expect(['!ASE\s*\n(.*\n){2}',
                                               'convergence NOT',
                                               'stopping'],
                                              timeout=self.timeout)
                        if i == 1:
                            raise SCFMaxIterationsError()
                        elif i == 2:
                            raise SCFConvergenceError()
                    except:
                        print('# Exception was thrown by pexpect.expect')
                        print(str(self.child))
                        self.child.logfile.flush()
                        self.stop()
                    self.child.logfile.flush()

            else:  # calculation == 'hund'
                self.site.runonly_perProcMpiExec(self.scratch,' pw.x -in pw.inp >>'+self.log)
                os.system("sed s/occupations.*/occupations=\\'fixed\\',/ <"+self.localtmp+"/pw.inp | sed s/ELECTRONS/ELECTRONS\\\\n\ \ startingwfc=\\'file\\',\\\\n\ \ startingpot=\\'file\\',/ | sed s/conv_thr.*/conv_thr="+num2str(self.conv_thr)+",/ | sed s/tot_magnetization.*/tot_magnetization="+num2str(self.totmag)+",/ >"+self.localtmp+"/pw2.inp")
                os.system(self.site.perHostMpiExec+' cp '+self.localtmp+'/pw2.inp '+self.scratch)
                self.cinp, self.cout = self.site.do_perProcMpiExec(self.scratch,'pw.x '+self.parflags+' -in pw2.inp')

        else:  # interactive
            self.localtmp.chdir()

            pwinp = self.localtmp.joinpath('pw.inp')
            Path.copy(pwinp, self.scratch)
            command = ' '.join(['pw.x', '-in', 'pw.inp'])

            if not self._spawned:
                self.child = pexpect.spawn(command)
                self._spawned = True
                self.child.logfile = self.logfile
                self.child.logfile.write(self.get_output_header().encode('utf-8'))
                try:
                    i = self.child.expect(['!ASE', 'convergence NOT', 'stopping'],
                                          timeout=self.timeout)
                    if i == 1:
                        raise SCFMaxIterationsError()
                    elif i == 2:
                        raise SCFConvergenceError()
                except:
                    print('# Exception was thrown by pexpect.expect')
                    print(str(self.child))
                    self.child.logfile.flush()
                    self.stop()

            else:  # QE process is already spawned

                self.child.send('C\n')
                for atom in self.atoms:
                    self.child.send('{0:25.14e} {1:25.14e} {2:25.10e}\n'.format(atom.x, atom.y, atom.z))
                self.child.expect('!ASE', timeout=self.timeout)

    def stop(self):

        if self._spawned:
            self.child.send('Q\n')

        self.child.logfile.close()
    
    def todict(self):
        return Espresso.todict(self)

espresso = iEspresso
