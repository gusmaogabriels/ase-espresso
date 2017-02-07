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


atomic_configs = [
    # 1 H: 1s1
    ([1, 0, 1],),
    # 2 He: 1s2
    ([1, 0, 2],),
    # 3 Li: 1s2 2s1
    ([1, 0, 2], [2, 0, 1]),
    # 4 Be: 1s2 2s2
    ([1, 0, 2], [2, 0, 2]),
    # 5 B: 1s2 2s2 2p1
    ([1, 0, 2], [2, 0, 2], [2, 1, 1]),
    # 6 C: 1s2 2s2 2p2
    ([1, 0, 2], [2, 0, 2], [2, 1, 2]),
    # 7 N: 1s2 2s2 2p3
    ([1, 0, 2], [2, 0, 2], [2, 1, 3]),
    # 8 O: 1s2 2s2 2p4
    ([1, 0, 2], [2, 0, 2], [2, 1, 4]),
    # 9 F: 1s2 2s2 2p5
    ([1, 0, 2], [2, 0, 2], [2, 1, 5]),
    # 10 Ne: 1s2 2s2 2p6
    ([1, 0, 2], [2, 0, 2], [2, 1, 6]),
    # 11 Na: [Ne] 3s1
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 1]),
    # 12 Mg: [Ne] 3s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2]),
    # 13 Al: [Ne] 3s2 3p1
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 1]),
    # 14 Si: [Ne] 3s2 3p2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 2]),
    # 15 P: [Ne] 3s2 3p3
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 3]),
    # 16 S: [Ne] 3s2 3p4
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 4]),
    # 17 Cl: [Ne] 3s2 3p5
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 5]),
    # 18 Ar: [Ne] 3s2 3p6
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6]),
    # 19 K: [Ar] 4s1
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [4, 0, 1]),
    # 20 Ca: [Ar] 4s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [4, 0, 2]),
    # 21 Sc: [Ar] 3d1 4s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 1], [4, 0, 2]),
    # 22 Ti: [Ar] 3d2 4s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 2], [4, 0, 2]),
    # 23 V: [Ar] 3d3 4s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 3], [4, 0, 2]),
    # 24 Cr: [Ar] 3d5 4s1
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 5], [4, 0, 1]),
    # 25 Mn: [Ar] 3d5 4s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 5], [4, 0, 2]),
    # 26 Fe: [Ar] 3d6 4s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 6], [4, 0, 2]),
    # 27 Co: [Ar] 3d7 4s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 7], [4, 0, 2]),
    # 28 Ni: [Ar] 3d8 4s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 8], [4, 0, 2]),
    # 29 Cu: [Ar] 3d10 4s1
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 1]),
    # 30 Zn: [Ar] 3d10 4s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2]),
    # 31 Ga: [Ar] 3d10 4s2 4p1
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 1]),
    # 32 Ge: [Ar] 3d10 4s2 4p2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 2]),
    # 33 As: [Ar] 3d10 4s2 4p3
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 3]),
    # 34 Se: [Ar] 3d10 4s2 4p4
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 4]),
    # 35 Br: [Ar] 3d10 4s2 4p5
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 5]),
    # 36 Kr: [Ar] 3d10 4s2 4p6
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6]),
    # 37 Rb: [Kr] 5s1
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [5, 0, 1]),
    # 38 Sr: [Kr] 5s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [5, 0, 2]),
    # 39 Y: [Kr] 4d1 5s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 1], [5, 0, 2]),
    # 40 Zr: [Kr] 4d2 5s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 2], [5, 0, 2]),
    # 41 Nb: [Kr] 4d4 5s1
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 4], [5, 0, 1]),
    # 42 Mo: [Kr] 4d5 5s1
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 5], [5, 0, 1]),
    # 43 Tc: [Kr] 4d5 5s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 5], [5, 0, 2]),
    # 44 Ru: [Kr] 4d7 5s1
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 7], [5, 0, 1]),
    # 45 Rh: [Kr] 4d8 5s1
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 8], [5, 0, 1]),
    # 46 Pd: [Kr] 4d10
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10]),
    # 47 Ag: [Kr] 4d10 5s1
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 1]),
    # 48 Cd: [Kr] 4d10 5s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2]),
    # 49 In: [Kr] 4d10 5s2 5p1
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 1]),
    # 50 Sn: [Kr] 4d10 5s2 5p2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 2]),
    # 51 Sb: [Kr] 4d10 5s2 5p3
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 3]),
    # 52 Te: [Kr] 4d10 5s2 5p4
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 4]),
    # 53 I: [Kr] 4d10 5s2 5p5
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 5]),
    # 54 Xe: [Kr] 4d10 5s2 5p6
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6]),
    # 55 Cs: [Xe] 6s1
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [6, 0, 1]),
    # 56 Ba: [Xe] 6s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [6, 0, 2]),
    # 57 La: [Xe] 5d1 6s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [5, 2, 1], [6, 0, 2]),
    # 58 Ce: [Xe] 4f1 5d1 6s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 1], [5, 2, 1], [6, 0, 2]),
    # 59 Pr: [Xe] 4f3 6s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 3], [6, 0, 2]),
    # 60 Nd: [Xe] 4f4 6s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 4], [6, 0, 2]),
    # 61 Pm: [Xe] 4f5 6s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 5], [6, 0, 2]),
    # 62 Sm: [Xe] 4f6 6s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 6], [6, 0, 2]),
    # 63 Eu: [Xe] 4f7 6s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 7], [6, 0, 2]),
    # 64 Gd: [Xe] 4f7 5d1 6s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 7], [5, 2, 1], [6, 0, 2]),
    # 65 Tb: [Xe] 4f9 6s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 9], [6, 0, 2]),
    # 66 Dy: [Xe] 4f10 6s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 10], [6, 0, 2]),
    # 67 Ho: [Xe] 4f11 6s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 11], [6, 0, 2]),
    # 68 Er: [Xe] 4f12 6s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 12], [6, 0, 2]),
    # 69 Tm: [Xe] 4f13 6s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 13], [6, 0, 2]),
    # 70 Yb: [Xe] 4f14 6s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 14], [6, 0, 2]),
    # 71 Lu: [Xe] 4f14 5d1 6s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 14], [5, 2, 1], [6, 0, 2]),
    # 72 Hf: [Xe] 4f14 5d2 6s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 14], [5, 2, 2], [6, 0, 2]),
    # 73 Ta: [Xe] 4f14 5d3 6s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 14], [5, 2, 3], [6, 0, 2]),
    # 74 W: [Xe] 4f14 5d4 6s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 14], [5, 2, 4], [6, 0, 2]),
    # 75 Re: [Xe] 4f14 5d5 6s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 14], [5, 2, 5], [6, 0, 2]),
    # 76 Os: [Xe] 4f14 5d6 6s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 14], [5, 2, 6], [6, 0, 2]),
    # 77 Ir: [Xe] 4f14 5d7 6s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 14], [5, 2, 7], [6, 0, 2]),
    # 78 Pt: [Xe] 4f14 5d9 6s1
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 14], [5, 2, 9], [6, 0, 1]),
    # 79 Au: [Xe] 4f14 5d10 6s1
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 14], [5, 2, 10], [6, 0, 1]),
    # 80 Hg: [Xe] 4f14 5d10 6s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 14], [5, 2, 10], [6, 0, 2]),
    # 81 Tl: [Xe] 4f14 5d10 6s2 6p1
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 14], [5, 2, 10], [6, 0, 2], [6, 1, 1]),
    # 82 Pb: [Xe] 4f14 5d10 6s2 6p2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 14], [5, 2, 10], [6, 0, 2], [6, 1, 2]),
    # 83 Bi: [Xe] 4f14 5d10 6s2 6p3
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 14], [5, 2, 10], [6, 0, 2], [6, 1, 3]),
    # 84 Po: [Xe] 4f14 5d10 6s2 6p4
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 14], [5, 2, 10], [6, 0, 2], [6, 1, 4]),
    # 85 At: [Xe] 4f14 5d10 6s2 6p5
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 14], [5, 2, 10], [6, 0, 2], [6, 1, 5]),
    # 86 Rn: [Xe] 4f14 5d10 6s2 6p6
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 14], [5, 2, 10], [6, 0, 2], [6, 1, 6]),
    # 87 Fr: [Rn] 7s1
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 14], [5, 2, 10], [6, 0, 2], [6, 1, 6], [7, 0, 1]),
    # 88 Ra: [Rn] 7s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 14], [5, 2, 10], [6, 0, 2], [6, 1, 6], [7, 0, 2]),
    # 89 Ac: [Rn] 6d1 7s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 14], [5, 2, 10], [6, 0, 2], [6, 1, 6], [6, 2, 1], [7, 0, 2]),
    # 90 Th: [Rn] 6d2 7s2
    ([1, 0, 2], [2, 0, 2], [2, 1, 6], [3, 0, 2], [3, 1, 6], [3, 2, 10], [4, 0, 2], [4, 1, 6], [4, 2, 10], [5, 0, 2], [5, 1, 6], [4, 3, 14], [5, 2, 10], [6, 0, 2], [6, 1, 6], [6, 2, 2], [7, 0, 2])
    ]

atomic_configs_dict = {
    'H':  ( '', [ '1s' ], 1, 'Hydrogen' ),
    'He': ( '', [ '1s2' ], 2, 'Helium' ),
    'Li': ( '', [ '1s2', '2s' ], 3, 'Lithium' ),
    'Be': ( '', [ '1s2', '2s2' ], 4, 'Beryllium' ),
    'B':  ( '', [ '1s2', '2s2', '2p' ], 5, 'Boron' ),
    'C':  ( '', [ '1s2', '2s2', '2p2' ], 6, 'Carbon' ),
    'N':  ( '', [ '1s2', '2s2', '2p3' ], 7, 'Nitrogen' ),
    'O':  ( '', [ '1s2', '2s2', '2p4' ], 8, 'Oxygen' ),
    'F':  ( '', [ '1s2', '2s2', '2p5' ], 9, 'Fluorine' ),
    'Ne': ( '', [ '1s2', '2s2', '2p6' ], 10, 'Neon' ),
    'Na': ( 'Ne', [ '3s' ], 11, 'Sodium' ),
    'Mg': ( 'Ne', [ '3s2' ], 12, 'Magnesium' ),
    'Al': ( 'Ne', [ '3s2', '3p' ], 13, 'Aluminum' ),
    'Si': ( 'Ne', [ '3s2', '3p2' ], 14, 'Silicon' ),
    'P':  ( 'Ne', [ '3s2', '3p3' ], 15, 'Phosphorus' ),
    'S':  ( 'Ne', [ '3s2', '3p4' ], 16, 'Sulfur' ),
    'Cl': ( 'Ne', [ '3s2', '3p5' ], 17, 'Chlorine' ),
    'Ar': ( 'Ne', [ '3s2', '3p6' ], 18, 'Argon' ),
    'K':  ( 'Ar', [ '4s' ], 19, 'Potassium' ),
    'Ca': ( 'Ar', [ '4s2' ], 20, 'Calcium' ),
    'Sc': ( 'Ar', [ '3d', '4s2' ], 21, 'Scandium' ),
    'Ti': ( 'Ar', [ '3d2', '4s2' ], 22, 'Titanium' ),
    'V':  ( 'Ar', [ '3d3', '4s2' ], 23, 'Vanadium' ),
    'Cr': ( 'Ar', [ '3d5', '4s' ], 24, 'Chromium' ),
    'Mn': ( 'Ar', [ '3d5', '4s2' ], 25, 'Manganese' ),
    'Fe': ( 'Ar', [ '3d6', '4s2' ], 26, 'Iron' ),
    'Co': ( 'Ar', [ '3d7', '4s2' ], 27, 'Cobalt' ),
    'Ni': ( 'Ar', [ '3d8', '4s2' ], 28, 'Nickel' ),
    'Cu': ( 'Ar', [ '3d10', '4s' ], 29, 'Copper' ),
    'Zn': ( 'Ar', [ '3d10', '4s2' ], 30, 'Zinc' ),
    'Ga': ( 'Ar', [ '3d10', '4s2', '4p' ], 31, 'Gallium' ),
    'Ge': ( 'Ar', [ '3d10', '4s2', '4p2' ], 32, 'Germanium' ),
    'As': ( 'Ar', [ '3d10', '4s2', '4p3' ], 33, 'Arsenic' ),
    'Se': ( 'Ar', [ '3d10', '4s2', '4p4' ], 34, 'Selenium' ),
    'Br': ( 'Ar', [ '3d10', '4s2', '4p5' ], 35, 'Bromine' ),
    'Kr': ( 'Ar', [ '3d10', '4s2', '4p6' ], 36, 'Krypton' ),
    'Rb': ( 'Kr', [ '5s' ], 37, 'Rubidium' ),
    'Sr': ( 'Kr', [ '5s2' ], 38, 'Strontium' ),
    'Y':  ( 'Kr', [ '4d', '5s2' ], 39, 'Yttrium' ),
    'Zr': ( 'Kr', [ '4d2', '5s2' ], 40, 'Zirconium' ),
    'Nb': ( 'Kr', [ '4d4', '5s' ], 41, 'Niobium' ),
    'Mo': ( 'Kr', [ '4d5', '5s' ], 42, 'Molybdenum' ),
    'Tc': ( 'Kr', [ '4d5', '5s2' ], 43, 'Technetium' ),
    'Ru': ( 'Kr', [ '4d7', '5s' ], 44, 'Ruthenium' ),
    'Rh': ( 'Kr', [ '4d8', '5s' ], 45, 'Rhodium' ),
    'Pd': ( 'Kr', [ '4d10' ], 46, 'Palladium' ),
    'Ag': ( 'Kr', [ '4d10', '5s' ], 47, 'Silver' ),
    'Cd': ( 'Kr', [ '4d10', '5s2' ], 48, 'Cadmium' ),
    'In': ( 'Kr', [ '4d10', '5s2', '5p' ], 49, 'Indium' ),
    'Sn': ( 'Kr', [ '4d10', '5s2', '5p2' ], 50, 'Tin' ),
    'Sb': ( 'Kr', [ '4d10', '5s2', '5p3' ], 51, 'Antimony' ),
    'Te': ( 'Kr', [ '4d10', '5s2', '5p4' ], 52, 'Tellurium' ),
    'I':  ( 'Kr', [ '4d10', '5s2', '5p5' ], 53, 'Iodine' ),
    'Xe': ( 'Kr', [ '4d10', '5s2', '5p6' ], 54, 'Xenon' ),
    'Cs': ( 'Xe', [ '6s' ], 55, 'Cesium' ),
    'Ba': ( 'Xe', [ '6s2' ], 56, 'Barium' ),
    'La': ( 'Xe', [ '5d', '6s2' ], 57, 'Lanthanum' ),
    'Ce': ( 'Xe', [ '4f', '5d', '6s2' ], 58, 'Cerium' ),
    'Pr': ( 'Xe', [ '4f3', '6s2' ], 59, 'Praseodymium' ),
    'Nd': ( 'Xe', [ '4f4', '6s2' ], 60, 'Neodymium' ),
    'Pm': ( 'Xe', [ '4f5', '6s2' ], 61, 'Promethium' ),
    'Sm': ( 'Xe', [ '4f6', '6s2' ], 62, 'Samarium' ),
    'Eu': ( 'Xe', [ '4f7', '6s2' ], 63, 'Europium' ),
    'Gd': ( 'Xe', [ '4f7', '5d', '6s2' ], 64, 'Gadolinium' ),
    'Tb': ( 'Xe', [ '4f9', '6s2' ], 65, 'Terbium' ),
    'Dy': ( 'Xe', [ '4f10', '6s2' ], 66, 'Dysprosium' ),
    'Ho': ( 'Xe', [ '4f11', '6s2' ], 67, 'Holmium' ),
    'Er': ( 'Xe', [ '4f12', '6s2' ], 68, 'Erbium' ),
    'Tm': ( 'Xe', [ '4f13', '6s2' ], 69, 'Thulium' ),
    'Yb': ( 'Xe', [ '4f14', '6s2' ], 70, 'Ytterbium' ),
    'Lu': ( 'Xe', [ '4f14', '5d', '6s2' ], 71, 'Lutetium' ),
    'Hf': ( 'Xe', [ '4f14', '5d2', '6s2' ], 72, 'Hafnium' ),
    'Ta': ( 'Xe', [ '4f14', '5d3', '6s2' ], 73, 'Tantalum' ),
    'W':  ( 'Xe', [ '4f14', '5d4', '6s2' ], 74, 'Tungsten' ),
    'Re': ( 'Xe', [ '4f14', '5d5', '6s2' ], 75, 'Rhenium' ),
    'Os': ( 'Xe', [ '4f14', '5d6', '6s2' ], 76, 'Osmium' ),
    'Ir': ( 'Xe', [ '4f14', '5d7', '6s2' ], 77, 'Iridium' ),
    'Pt': ( 'Xe', [ '4f14', '5d9', '6s' ], 78, 'Platinum' ),
    'Au': ( 'Xe', [ '4f14', '5d10', '6s' ], 79, 'Gold' ),
    'Hg': ( 'Xe', [ '4f14', '5d10', '6s2' ], 80, 'Mercury' ),
    'Tl': ( 'Xe', [ '4f14', '5d10', '6s2', '6p' ], 81, 'Thallium' ),
    'Pb': ( 'Xe', [ '4f14', '5d10', '6s2', '6p2' ], 82, 'Lead' ),
    'Bi': ( 'Xe', [ '4f14', '5d10', '6s2', '6p3' ], 83, 'Bismuth' ),
    'Po': ( 'Xe', [ '4f14', '5d10', '6s2', '6p4' ], 84, 'Polonium' ),
    'At': ( 'Xe', [ '4f14', '5d10', '6s2', '6p5' ], 85, 'Astatine' ),
    'Rn': ( 'Xe', [ '4f14', '5d10', '6s2', '6p6' ], 86, 'Radon' ),
    'Fr': ( 'Rn', [ '7s' ], 87, 'Francium' ),
    'Ra': ( 'Rn', [ '7s2' ], 88, 'Radium' ),
    'Ac': ( 'Rn', [ '6d', '7s2' ], 89, 'Actinium' ),
    'Th': ( 'Rn', [ '6d2', '7s2' ], 90, 'Thorium' )
    }


ldict = {'s': 0, 'p': 1, 'd': 2, 'f': 3}


def hundmagperchannel(channel):
    """
    returns the magnetization according to Hund for a channel
    in the form of e.g. '5d8'
    """
    l = ldict[channel[1]]
    if len(channel)==2:
        nelec = 1
    else:
        nelec = int(channel[2:])

    nstates = 2*l+1
    if nelec>nstates:
        return 2*nstates-nelec
    else:
        return nelec

def hundmag(s):
    """
    returns the magnetization according to Hund
    s can either be an atomic symbol or a list of valence states,
    e.g. ['4f7', '5d', '6s2']
    """
    mag = 0
    if hasattr(s, 'upper'):
        configs = atomic_configs_dict[s][1]
    else:
        configs = s
    for c in configs:
        mag += hundmagperchannel(c)
    return mag
