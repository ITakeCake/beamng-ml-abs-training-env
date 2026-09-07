"""Co-simulation UDP link: BeamNG.tech's physics-rate coupling, driven from Python.

Replaces beamngpy's frame-gated step() loop. BeamNG runs a vehicle controller
(tech/cosimulationCoupling) that, each exchange, sends an ordered "To" packet of
signals and applies an ordered "From" packet of per-wheel brake torques. The
controller negates braking torque, so send NEGATIVE to brake.

Wire format: raw little-endian float64 array. First value in every packet is an
incrementing id; BeamNG rejects an inbound id <= the last it saw.
"""
import os
import socket
import struct
import time

DIAG_FILE = os.path.join(os.getcwd(), "diag_latest.log")
_diag_file = DIAG_FILE


def configure_diag_file(path, create=True):
    """Route diagnostics to a new run-local file without truncating old logs."""
    global _diag_file
    destination = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if create:
        with open(destination, "x", encoding="utf-8"):
            pass
    _diag_file = destination
    return destination


def diag_write(msg):
    print("DIAG " + msg, flush=True)
    try:
        with open(_diag_file, "a") as f:
            f.write("DIAG " + msg + "\n")
    except OSError:
        pass

SEND_IP, SEND_PORT = "127.0.0.1", 64890      # BeamNG -> Python
RECV_IP, RECV_PORT = "127.0.0.1", 64891      # Python -> BeamNG


def _lua_sig_to(sigs):
    return ",".join("{groupName='%s',name='%s',type='number'}" % s for s in sigs)


def _lua_sig_from(sigs):
    tail = "type='number',isValue=true,isMultiply=false,isAdd=false,isFreeze=false"
    return ",".join("{groupName='%s',name='%s',%s}" % (g, n, tail) for g, n in sigs)


class CoSimLink:
    """Owns the sockets + the signal contract. One exchange = send torques, get obs."""

    def __init__(self, sig_to, sig_from, time_3rd_party=0.005, timeout=3.0):
        self.sig_to = sig_to
        self.sig_from = sig_from
        self.to_n = len(sig_to)
        self.from_n = len(sig_from)
        self.time_3rd_party = time_3rd_party
        self._timeout = timeout
        self._rx = None
        self._tx = None
        self._out_id = 1
        # Timing split: recv-block (BeamNG send interval) vs gap (Python/SB3 side).
        self._t_prev_ret = None
        self._sum_wait = self._sum_gap = 0.0
        self._n_timed = 0

    def load_cmd(self):
        """Lua for the vehicle VM: build the config table and load the controller."""
        cdata = (
            "signalsTo={%s},signalsFrom={%s},"
            "sensorMap={IMUs={},GPSs={},idealRADARs={},roads={}},"
            "time3rdParty=%g,pingTime=1e-5,"
            "udpSendIP='%s',udpSendPort=%d,udpReceiveIP='%s',udpReceivePort=%d,"
            "enableVSL=false,enableCosim=true"
        ) % (_lua_sig_to(self.sig_to), _lua_sig_from(self.sig_from),
             self.time_3rd_party, SEND_IP, SEND_PORT, RECV_IP, RECV_PORT)
        return ("local lpack=require('lpack');local cData={%s};"
                "controller.loadControllerExternal("
                "'tech/cosimulationCoupling','cosimulationCoupling',"
                "lpack.encode({cData}))") % cdata

    @staticmethod
    def stop_cmd():
        return ("controller.getController('cosimulationCoupling').stop();"
                "controller.unloadControllerExternal('cosimulationCoupling')")

    def open(self):
        self._rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rx.bind((SEND_IP, SEND_PORT))
        self._rx.settimeout(self._timeout)
        self._tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._out_id = 1

    def drain(self):
        """Throw away datagrams queued while Python was not reading.

        BeamNG transmits every control period whether or not anyone answers, so
        an optimizer pause leaves a backlog in the socket buffer, at 400 Hz a
        half-second PPO update is ~200 packets. The next episode would then be
        driven by the previous episode's dying moments: PPO-67 episode 2 opened
        on a packet reading 0.32 m/s while the car was actually doing 40, filled
        the 64-frame history with it, and braked at 0.559 g where episode 1 (an
        empty queue) managed 1.186 g with identical weights.

        Returns the number of stale packets discarded.
        """
        if self._rx is None:
            return 0
        dropped = 0
        self._rx.setblocking(False)
        try:
            while True:
                try:
                    self._rx.recvfrom(2048)
                except (BlockingIOError, socket.timeout):
                    break
                except OSError:
                    break
                dropped += 1
        finally:
            self._rx.settimeout(self._timeout)
        return dropped

    def exchange(self, from_values):
        """Block for BeamNG's next packet, reply with from_values. Returns the
        decoded "To" values (id stripped), or None on timeout."""
        t_enter = time.perf_counter()
        if self._t_prev_ret is not None:                # time spent OUTSIDE exchange
            self._sum_gap += t_enter - self._t_prev_ret
        try:
            data, _ = self._rx.recvfrom(2048)
        except socket.timeout:
            return None
        t_recv = time.perf_counter()
        self._sum_wait += t_recv - t_enter              # blocked on BeamNG's packet
        self._n_timed += 1
        if self._n_timed >= 200:
            diag_write("TIMING n=%d recv_wait=%.1fms outside=%.1fms (of %.1fms total)"
                       % (self._n_timed, 1000 * self._sum_wait / self._n_timed,
                          1000 * self._sum_gap / self._n_timed,
                          1000 * (self._sum_wait + self._sum_gap) / self._n_timed))
            self._sum_wait = self._sum_gap = 0.0
            self._n_timed = 0
        if len(data) != (self.to_n + 1) * 8:
            raise ValueError("cosim packet %d bytes, expected %d"
                             % (len(data), (self.to_n + 1) * 8))
        vals = struct.unpack("<%dd" % (self.to_n + 1), data)
        self._tx.sendto(
            struct.pack("<%dd" % (self.from_n + 1), self._out_id, *from_values),
            (RECV_IP, RECV_PORT))
        self._out_id += 1
        self._t_prev_ret = time.perf_counter()
        return vals[1:]

    def close(self):
        for s in (self._rx, self._tx):
            try:
                if s:
                    s.close()
            except Exception:
                pass
        self._rx = self._tx = None
