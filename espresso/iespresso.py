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

import os
import socket
from subprocess import Popen

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
import io
try:
    from path import Path
except:
    from pypath import Path
import logging
import inspect
import pexpect

from ase.calculators.calculator import (Calculator, all_changes,
                                        PropertyNotImplementedError)
import ase.units as units
from ase.calculators.socketio import  SocketIOCalculator

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

class SCFTimeoutError(Exception):
    pass

def actualunixsocketname(name):
    return '/tmp/ipi_{}'.format(name)


class SocketClosed(OSError):
    pass


from .espresso import Espresso

def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]

class IPIProtocol:
    """Communication using IPI protocol."""

    def __init__(self, socket, txt=None):
        self.socket = socket
        self.firstrun = True

        if txt is None:
            def log(*args):
                pass
        else:
            def log(*args):
                print('Driver:', *args, file=txt)
                txt.flush()
        self.log = log

    def sendmsg(self, msg):
        self.log('  sendmsg', repr(msg))
        # assert msg in self.statements, msg
        msg = msg.encode('ascii').ljust(12)
        self.socket.sendall(msg)

    def _recvall(self, nbytes):
        """Repeatedly read chunks until we have nbytes.

        Normally we get all bytes in one read, but that is not guaranteed."""
        remaining = nbytes
        chunks = []
        while remaining > 0:
            chunk = self.socket.recv(remaining)
            if len(chunk) == 0:
                # (If socket is still open, recv returns at least one byte)
                raise SocketClosed()
            chunks.append(chunk)
            remaining -= len(chunk)
        msg = b''.join(chunks)
        assert len(msg) == nbytes and remaining == 0
        return msg

    def recvmsg(self):
        msg = self._recvall(12)
        if not msg:
            raise SocketClosed()

        assert len(msg) == 12, msg
        msg = msg.rstrip().decode('ascii')
        # assert msg in self.responses, msg
        self.log('  recvmsg', repr(msg))
        return msg

    def send(self, a, dtype):
        buf = np.asarray(a, dtype).tobytes()
        # self.log('  send {}'.format(np.array(a).ravel().tolist()))
        self.log('  send {} bytes of {}'.format(len(buf), dtype))
        self.socket.sendall(buf)

    def recv(self, shape, dtype):
        a = np.empty(shape, dtype)
        nbytes = np.dtype(dtype).itemsize * np.prod(shape)
        buf = self._recvall(nbytes)
        assert len(buf) == nbytes, (len(buf), nbytes)
        self.log('  recv {} bytes of {}'.format(len(buf), dtype))
        a.flat[:] = np.frombuffer(buf, dtype=dtype)
        # self.log('  recv {}'.format(a.ravel().tolist()))
        assert np.isfinite(a).all()
        return a

    def sendposdata(self, cell, icell, positions, properties):
        if self.firstrun or 'cell' in properties:
            assert cell.size == 9
            assert icell.size == 9
        assert positions.size % 3 == 0

        self.log(' sendposdata')
        self.sendmsg('POSDATA')
        if 'ensemble_energies' in properties:
            self.log(' genensemble')
        if self.firstrun or 'cell' in properties:
            self.send(cell.T / units.Bohr, np.float64)
            self.send(icell.T * units.Bohr, np.float64)
        self.send(len(positions), np.int32)
        self.send(positions / units.Bohr, np.float64)

    def recvposdata(self):
        cell = self.recv((3, 3), np.float64).T.copy()
        icell = self.recv((3, 3), np.float64).T.copy()
        natoms = self.recv(1, np.int32)
        natoms = int(natoms)
        positions = self.recv((natoms, 3), np.float64)
        return cell * units.Bohr, icell / units.Bohr, positions * units.Bohr

    def sendrecv_force(self, properties):
        self.log(' sendrecv_force')
        results = []
        self.sendmsg('GETFORCE')
        msg = self.recvmsg()
        assert msg == 'FORCEREADY', msg
        e = self.recv(1, np.float64)[0]
        results += [e * units.Ha]
        natoms = self.recv(1, np.int32)
        assert natoms >= 0
        if 'forces' in properties:
            forces = self.recv((int(natoms), 3), np.float64)
            results += [(units.Ha / units.Bohr) * forces]
        else:
            forces = None
            results += [None]#[np.zeros((int(natoms), 3))]
        if 'stress' in properties:
            virial = self.recv((3, 3), np.float64).T.copy()
            results += [units.Ha * virial]
        else:
            virial = None
            results += [None]#[np.zeros((3, 3))]
        if 'ensemble_energies' in properties:
            energies = self.recv((2000,1), np.float64)
            beefxc = self.recv((32,1), np.float64)
            results += [[energies,beefxc]]
        else:
            results += [None]
        nmorebytes = self.recv(1, np.int32)
        nmorebytes = int(nmorebytes)
        if nmorebytes > 0:
            # Receiving 0 bytes will block forever on python2.
            morebytes = self.recv(nmorebytes, np.byte)
        else:
            morebytes = b''
        self.morebytes = morebytes
        return results

    def sendforce(self, energy, forces, virial,
                  morebytes=np.zeros(1, dtype=np.byte)):
        assert np.array([energy]).size == 1
        assert forces.shape[1] == 3
        assert virial.shape == (3, 3)

        self.log(' sendforce')
        self.sendmsg('FORCEREADY')  # mind the units
        self.send(np.array([energy / units.Ha]), np.float64)
        natoms = len(forces)
        self.send(np.array([natoms]), np.int32)
        self.send(units.Bohr / units.Ha * forces, np.float64)
        self.send(1.0 / units.Ha * virial.T, np.float64)
        # We prefer to always send at least one byte due to trouble with
        # empty messages.  Reading a closed socket yields 0 bytes
        # and thus can be confused with a 0-length bytestring.
        self.send(np.array([len(morebytes)]), np.int32)
        self.send(morebytes, np.byte)

    def status(self):
        self.log(' status')
        self.sendmsg('STATUS')
        msg = self.recvmsg()
        return msg

    def end(self):
        self.log(' end')
        self.sendmsg('EXIT')

    def recvinit(self):
        self.log(' recvinit')
        bead_index = self.recv(1, np.int32)
        nbytes = self.recv(1, np.int32)
        initbytes = self.recv(nbytes, np.byte)
        return bead_index, initbytes

    def sendinit(self, properties):
        # XXX Not sure what this function is supposed to send.
        # It 'works' with QE, but for now we try not to call it.
        self.log(' sendinit')
        self.sendmsg('INIT')
        self.send(0, np.int32)  # 'bead index' always zero for now
        ### Here is a very hacky external controler for QE
        ### by Gabriel S. Gusmao : gusmaogabriels@gmail.com
        
        binstring = ''.join(['1' if _ in properties else '0' for _ in\
                             ['energy', 'forces', 'stress', 'cell', 'ensemble_energies']])
        
        hotint = int(binstring,2)+1 # adding one to make it ASE compliant
        self.send(hotint, np.int32)  # action enconded integer
        ### ENDOFHACK
        # We send one byte, which is zero, since things may not work
        # with 0 bytes.  Apparently implementations ignore the
        # initialization string anyway.
        #self.send(1, np.int32)
        self.send(np.zeros(1), np.byte)  # initialization string

    def ionic_step(self, positions, cell, properties):
        if self.firstrun or 'cell' in properties:
            icell = np.linalg.pinv(cell).transpose()
        else:
            icell = None
        self.sendposdata(cell, icell, positions, properties)
        msg = self.status()
        assert msg == 'HAVEDATA', msg
        res = self.sendrecv_force(properties)
        r = dict(filter(lambda _ : id(_[1])!=id(None) ,\
                  zip(['energy','forces','virial','ensemble_energies'],res)))
        return r

    def calculate(self, positions, cell, properties):
        self.log('calculate')
        msg = self.status()
        # We don't know how NEEDINIT is supposed to work, but some codes
        # seem to be okay if we skip it and send the positions instead.
        if msg == 'NEEDINIT':
            self.sendinit(properties)
            msg = self.status()
        assert msg == 'READY', msg
        r = self.ionic_step(positions, cell, properties)
        self.firstrun = False
        return r

class SocketServer:
    default_port = 31415

    def __init__(self, client_command=None, port=None,
                 unixsocket=None, timeout=None, cwd=None, 
                 log=None,properties=['energy']):
        """Create server and listen for connections.

        Parameters:

        client_command: Shell command to launch client process, or None
            The process will be launched immediately, if given.
            Else the user is expected to launch a client whose connection
            the server will then accept at any time.
            One calculate() is called, the server will block to wait
            for the client.
        port: integer or None
            Port on which to listen for INET connections.  Defaults
            to 31415 if neither this nor unixsocket is specified.
        unixsocket: string or None
            Filename for unix socket.
        timeout: float or None
            timeout in seconds, or unlimited by default.
            This parameter is passed to the Python socket object; see
            documentation therof
        log: file object or None
            useful debug messages are written to this."""

        if unixsocket is None and port is None:
            port = self.default_port
        elif unixsocket is not None and port is not None:
            raise ValueError('Specify only one of unixsocket and port')

        self.port = port
        self.unixsocket = unixsocket
        self.timeout = timeout
        self._closed = False
        self._created_socket_file = None  # file to be unlinked in close()

        if unixsocket is not None:
            self.serversocket = socket.socket(socket.AF_UNIX)
            actualsocket = actualunixsocketname(unixsocket)
            try:
                self.serversocket.bind(actualsocket)
            except OSError as err:
                raise OSError('{}: {}'.format(err, repr(actualsocket)))
            self._created_socket_file = actualsocket
            conn_name = 'UNIX-socket {}'.format(actualsocket)
        else:
            self.serversocket = socket.socket(socket.AF_INET,  socket.SOCK_STREAM)
            self.serversocket.setsockopt(socket.SOL_SOCKET,
                                         socket.SO_REUSEADDR, 1)
            self.serversocket.bind(('', port))
            conn_name = 'INET port {}'.format(port)
            if log:
                print(conn_name, get_ip_address(),file=log)
                print('SOCKET',self._created_socket_file,socket.gethostname(),id(self),file=log)

        if log:
            print('Accepting clients on {}'.format(conn_name), file=log)

        self.serversocket.settimeout(timeout)

        self.serversocket.listen(1)

        self.log = log

        self.proc = None

        self.protocol = None
        self.clientsocket = None
        self.address = None
        self.cwd = cwd

        if client_command is not None:
            client_command = client_command.format(port=port,
                                                   unixsocket=unixsocket)
            if log:
                print('Launch subprocess: {}'.format(client_command), file=log)
            print('CMD',client_command)
            self.proc = Popen(client_command, shell=True,
                              cwd=self.cwd)
            # self._accept(process_args)
            #self._accept()

    def _accept(self, client_command=None):
        """Wait for client and establish connection."""
        # It should perhaps be possible for process to be launched by user
        log = self.log
        if self.log:
            print('Awaiting client', file=self.log)
        print(id(self),'Awaiting client')
        # If we launched the subprocess, the process may crash.
        # We want to detect this, using loop with timeouts, and
        # raise an error rather than blocking forever.
        if self.proc is not None:
            self.serversocket.settimeout(1.0)

        while True:
            try:
                self.clientsocket, self.address = self.serversocket.accept()
                print('-'*10+'>',self.clientsocket, self.address)
            except socket.timeout:
                if self.proc is not None:
                    status = self.proc.poll()
                    if status is not None:
                        raise OSError('Subprocess terminated unexpectedly'
                                      ' with status {}'.format(status))
            else:
                break

        self.serversocket.settimeout(self.timeout)
        self.clientsocket.settimeout(self.timeout)

        if log:
            # For unix sockets, address is b''.
            source = ('client' if self.address == b'' else self.address)
            print('Accepted connection from {}'.format(source), file=log)

        self.protocol = IPIProtocol(self.clientsocket, txt=log)

    def close(self):
        if self._closed:
            return

        if self.log:
            print('Close socket server', file=self.log)
        self._closed = True

        # Proper way to close sockets?
        # And indeed i-pi connections...
        if self.protocol is not None:
             self.protocol.end()  # Send end-of-communication string
        self.protocol = None
        if self.clientsocket is not None:
            self.clientsocket.close()  # shutdown(socket.SHUT_RDWR)
        if self.proc is not None:
            exitcode = self.proc.wait()
            if exitcode != 0:
                import warnings
                # Quantum Espresso seems to always exit with status 128,
                # even if successful.
                # Should investigate at some point
                warnings.warn('Subprocess exited with status {}'
                              .format(exitcode))
        if self.serversocket is not None:
            self.serversocket.close()
        if self._created_socket_file is not None:
            assert self._created_socket_file.startswith('/tmp/ipi_')
            os.unlink(self._created_socket_file)
        # self.log('IPI server closed')

    def calculate(self, atoms, properties):
        """Send geometry to client and return calculated things as dict.

        This will block until client has established connection, then
        wait for the client to finish the calculation."""
        assert not self._closed

        # If we have not established connection yet, we must block
        # until the client catches up:
        if self.protocol is None:
            self._accept()
        return self.protocol.calculate(atoms.positions, atoms.cell, properties)


class iEspresso(Espresso):
    implemented_properties = ['energy', 'forces', 'stress', 'cell', 'ensemble_energies']
    supported_changes = {'positions', 'numbers', 'cell', 'pbc',
               'initial_charges', 'initial_magmoms'}

    ## -> remastered from SocketIOCalculator

    def __init__(self, atoms, port=None, socket_type='UNIX',
                 unixsocket=None, timeout=None, log=None, ionode_address=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialize socket I/O calculator.

        This calculator launches a server which passes atomic
        coordinates and unit cells to an external code via a socket,
        and receives energy, forces, and stress in return.

        ASE integrates this with the Quantum Espresso, FHI-aims and
        Siesta calculators.  This works with any external code that
        supports running as a client over the i-PI protocol.

        Parameters:

        calc: calculator or None

            If calc is not None, a client process will be launched
            using calc.command, and the input file will be generated
            using ``calc.write_input()``.  Otherwise only the server will
            run, and it is up to the user to launch a compliant client
            process.

        port: integer

            port number for socket.  Should normally be between 1025
            and 65535.  Typical ports for are 31415 (default) or 3141.

        unixsocket: str or None

            if not None, ignore host and port, creating instead a
            unix socket using this name prefixed with ``/tmp/ipi_``.
            The socket is deleted when the calculator is closed.

        timeout: float >= 0 or None

            timeout for connection, by default infinite.  See
            documentation of Python sockets.  For longer jobs it is
            recommended to set a timeout in case of undetected
            client-side failure.

        log: file object or None (default)

            logfile for communication over socket.  For debugging or
            the curious.

        In order to correctly close the sockets, it is
        recommended to use this class within a with-block:

        >>> with SocketIOCalculator(...) as calc:
        ...    atoms.calc = calc
        ...    atoms.get_forces()
        ...    atoms.rattle()
        ...    atoms.get_forces()

        It is also possible to call calc.close() after
        use.  This is best done in a finally-block."""
        self.timeout = timeout
        self.server = None
        if isinstance(log, str):
            self.socket_log = open(log, 'w')
            self.socket_log_was_opened = True
        else:
            self.socket_log = log
            self.socket_log_was_opened = False

        # We only hold these so we can pass them on to the server.
        # They may both be None as stored here.
        self._port = port
        self._unixsocket = unixsocket
        self._socket_type = socket_type
        
        # First time calculate() is called, system_changes will be
        # all_changes.  After that, only positions and cell may change.
        self.calculator_initialized = False
        self.ionode_address = ionode_address
        self.atoms = atoms.copy()
        atoms.calc = self 
        atoms.get_ensemble_energies = self.get_ensemble_energies
        
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
        
    def launch_server(self, cmd=None, properties=['energy']):
        if not isinstance(self.directory,str):
            cwd = self.directory.joinpath('')
        else:
            cwd = self.directory
        self.server = SocketServer(client_command=cmd,
                                   unixsocket=self._unixsocket,
                                   port=self._port,
                                   timeout=self.timeout, log=self.socket_log,
                                   cwd=cwd)

    def update(self, atoms, properties=['energy']):
        '''
        Check if the atoms object has changes and perform a calculation
        when it does
        '''
        from ase.calculators.calculator import compare_atoms
        
        state = self.check_state(atoms)
        if any([_ in state for _ in ['positions','cell']]):
            self.results = {}
            if 'energy' in state and 'energy' not in properties:
                properties += ['energy']
            if 'cell' in state and 'cell' not in properties:
                properties += ['cell']
        if all([_ not in properties for _ in ['forces','ensemble_energies']]):
            properties += ['forces']
        self.atoms = atoms.copy()
 
        if self.atoms is None:
            self.set_atoms(atoms)
        if self.calculation_required(atoms, properties):
            self.calculate(atoms, properties)
            self.recalculate = False

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        
        bad = [change for change in system_changes
               if change not in self.supported_changes]

        if self.calculator_initialized and any(bad):
            raise PropertyNotImplementedError(
                'Cannot change {} through IPI protocol.  '
                'Please create new socket calculator.'
                .format(bad if len(bad) > 1 else bad[0]))

        self.calculator_initialized = True

        if self.server is None:
            #if 'vc' in self.calculation:
            #    self.cell_factor = 10
            #self.calculation = 'scf'
            self.cell_dynamics = 'ipi'
            self.ion_dynamics = 'ipi'
            self.dontcalcforces = False
                    
            self.write_input(atoms, properties=properties,
                                  system_changes=system_changes)
            
            if self._socket_type=='UNIX' or ((not self._unixsocket and not self._port) and self._socket_type!='INET'):
                self._socket_type = 'UNIX'
                if not self._unixsocket:
                    self._unixsocket = self.scratch.split('/')[-1]
                socket_string = ' --ipi {0}:UNIX >> {1}'.format(self._unixsocket,self.log)
            elif self._socket_type=='INET' or (not self._unixsocket and self._port):
                self._socket_type = 'INET'
                port = SocketServer.default_port
                SocketServer.default_port += 1
                while socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect_ex(('localhost', port)) == 0:
                    port += 1
                self._port = port
                if re.match('^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$', self.ionode_address):
                    self._ip = self.ionode_address
                elif self.ionode_address in self.site.nic_inet_ips.keys():
                    self._ip = self.site.nic_inet_ips[self.ionode_address]
                else:
                    raise Exception('Not a valida IPV4 address or NIC interface: {}'.format(self.ionode_address))
                socket_string = ' --ipi {0}:{1} >> {2}'.format(self._ip,self._port,self.log)
            else:
                raise Exception('Socket type: {} not implemented.'.format(self._socket_type))
            if self.socket_log:
                print(self._socket_type,file=self.socket_log)
            cmd = ' '.join(self.command) + socket_string

            self.launch_server(cmd)
        results = self.server.calculate(atoms,properties)
        
        if 'virial' in results.keys():
            if self.atoms.number_of_lattice_vectors == 3 and any(self.atoms.pbc):
                from ase.constraints import full_3x3_to_voigt_6_stress
                vol = atoms.get_volume()
                results['stress'] = -full_3x3_to_voigt_6_stress(results['virial']) / vol
            else:
                raise Exception('Stress calculation: cell and periodic boundary conditions must be defined.')
        self.results.update(results)
    
    def close(self):
        if self.server is not None:
            self.server.close()
            self.server = None
            self.calculator_initialized = False
            if self.socket_log_was_opened:
                try:
                    self.log.close()
                except:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
    """
    def todict(self):
        d = {'type': 'calculator',
             'name': 'socket-driver'}
        if self.calc is not None:
            d['calc'] = self.calc.todict()
        return d
    """

    def get_ensemble_energies(self):
        # Bypassing the standard ASE atoms and calculator for ASE folks
        # opted not to include get_ensemble_energies as a default method
        self.update(self.atoms,['ensemble_energies'])
        if self.server:
            return self.results['ensemble_energies']
        else:
            return self.get_nonselfconsistent_energies(self, type='beefvdw')
        

iespresso = iEspresso
