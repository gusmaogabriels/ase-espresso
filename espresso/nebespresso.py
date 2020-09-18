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

from __future__ import print_function, absolute_import

import copy
import threading

from ase.neb import NEB

from .siteconfig import SiteConfig


__version__ = '0.3.3'


def splitinto(l, n):
    '''
    Split a list into `n` sublists of roughly equal size

    Args:
        l (`list`) :
            list to be split
        n (int) :
            number of sublists
    '''

    q, r = divmod(len(l), n)
    indices = [q * i + min(i, r) for i in range(n + 1)]
    return [l[indices[i]:indices[i + 1]] for i in range(n)]


class NEBEspresso(NEB):

    def __init__(self, images, site=None, outprefix='neb', **neb_kwargs):

        super().__init__(images,**neb_kwargs)

        self.site = site
        self.outprefix = outprefix
        self.jobs = []
        self.initialize()
        self.__hasservers__ = False

    @property
    def site(self):
        return self._site

    @site.setter
    def site(self, value):
        if value is None:
            self._site = SiteConfig.check_scheduler()
            if self._site.scheduler is None:
                raise NotImplementedError('Interactive NEB is not supported')
        else:
            self._site = value

    def wait_for_total_energies(self,properties=['energy','forces'],system_changes=['positions']):
        '''
        Calculate the energy for each thread in a separate thread and
        wait until all the calculations are finished.
        '''
        threads = [threading.Thread(target=self.images[i]._calc.calculate,
                                    args=(self.images[i],),
                                    kwargs={'properties':properties,'system_changes':system_changes})
                   for i in range(1, self.nimages - 1)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()


    def get_forces(self):

        #if not self.__hasservers__:
        #    [image._calc.calculate(image,properties=[],system_changes=[]) for image in self.images[1:-1]]
        #    self.__hasservers__ = True
        #    self.wait_for_total_energies(properties=[],system_changes=[])
        self.wait_for_total_energies()
        return super().get_forces()

    def initialize(self):
        'Create the calculator instances'

        imageprocs = splitinto(self.site.proclist, len(self.images) - 2)

        images = self.images[1:self.nimages - 1]

        for i, (image, procs) in enumerate(zip(images, imageprocs)):

            site = copy.deepcopy(self.site)

            site.proclist = procs
            site.nprocs = len(site.proclist)
            site.usehostfile = True

            image._calc.set(outdir='{0:s}_{1:04d}'.format(self.outprefix, i),
                            site=site)
        
        
        
