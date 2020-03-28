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

from __future__ import unicode_literals

from collections import namedtuple
import numpy as np
from ase import constraints


__version__ = '0.3.3'


speciestuple = namedtuple('speciestuple',
                          ['symbol', 'mass', 'magmom', 'U', 'J', 'U_alpha'])


def num2str(x):
    '''
    Add 'd00' to floating point number to avoid random trailing digits in
    Fortran input routines
    '''

    if 'e' in str(x):
        return str(x)
    else:
        return str(x) + 'd00'


def bool2str(x):
    'Convert python to fortran logical'

    if x:
        return '.true.'
    else:
        return '.false.'


def convert_constraints(atoms):
    '''
    Convert some of ase's constraints to pw.x constraints for pw.x internal
    relaxation returns constraints which are simply expressed as setting
    force components as first list and other contraints that are
    implemented in espresso as second list
    '''

    if atoms.constraints:
        n = len(atoms)
        if n == 0:
            return [], []
        forcefilter = []
        otherconstr = []
        for c in atoms.constraints:
            if isinstance(c, constraints.FixAtoms):
                if len(forcefilter) == 0:
                    forcefilter = np.ones((n, 3), np.int)
                forcefilter[c.index] = [0, 0, 0]
            elif isinstance(c, constraints.FixCartesian):
                if len(forcefilter) == 0:
                    forcefilter = np.ones((n, 3), np.int)
                forcefilter[c.a] = c.mask
            elif isinstance(c, constraints.FixBondLengths):
                for d in c.constraints:
                    otherconstr.append("'distance' %d %d" % (d.indices[0]+1,d.indices[1]+1))
            elif isinstance(c, constraints.FixInternals):
            # we ignore the epsilon in FixInternals because there can only be one global
            # epsilon be defined in espresso for all constraints
                for d in c.constraints:
                    if isinstance(d, constraints.FixInternals.FixBondLengthAlt):
                        otherconstr.append("'distance' %d %d %s" % (d.indices[0]+1,d.indices[1]+1,num2str(d.bond)))
                    elif isinstance(d, constraints.FixInternals.FixAngle):
                        otherconstr.append("'planar_angle' %d %d %d %s" % (d.indices[0]+1,d.indices[1]+1,d.indices[2]+1,num2str(np.arccos(d.angle)*180./np.pi)))
                    elif isinstance(d, constraints.FixInternals.FixDihedral):
                        otherconstr.append("'torsional_angle' %d %d %d %d %s" % (d.indices[0]+1,d.indices[1]+1,d.indices[2]+1,d.indices[3]+1,num2str(np.arccos(d.angle)*180./np.pi)))
                    else:
                        raise NotImplementedError('constraint {} from FixInternals not implemented\n'
                            'consider ase-based relaxation with this constraint instead'.format(d.__name__))
            else:
                raise NotImplementedError('constraint {} not implemented\n'
                    'consider ase-based relaxation with this constraint instead'.format(c.__name__))
        return forcefilter, otherconstr
    else:
        return [], []
