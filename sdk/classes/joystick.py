import threading, select
from collections import deque
from typing import List, Tuple
import time
import glob

from evdev import InputDevice, ecodes
from sdk.core.exceptions import JoystickEstopError, JoystickAPIError


class Joystick:
    def __init__(self, max_cmd: Tuple[float] | List[float]):
        self.idx      = {"ABS_Y": 0, "ABS_X": 1, "ABS_RY": 2, "ABS_RX": 3}
        self.max_cmd   = [1.0] * 4
        for i in range(min(len(max_cmd), len(self.max_cmd))):
            self.max_cmd[i] = max_cmd[i]
        self.dz       = 0.03
        self.dz_th     = [m*self.dz for m in self.max_cmd]
        self.scale    = [-self.max_cmd[0]/32767.0, self.max_cmd[1]/32767.0, -self.max_cmd[2]/32767.0, self.max_cmd[3]/32767.0]
        
        self.stick_input = [0.0] * 4
        self.btn_input = {"BTN_TL": 0, "BTN_TR": 0, "hatX": 0, "hatY": 0, "X": 0, "B": 0, "A": 0, "Y": 0, "ABS_Z": 0, "ABS_RZ": 0}
        self.robot_cmd = [0.0] * 6
        self.robot_prev_cmd = [0.0] * 6

        self.btn_alias = {
            "BTN_X": "X", "BTN_NORTH": "X",
            "BTN_B": "B", "BTN_EAST": "B",
            "BTN_A": "A", "BTN_GAMEPAD": "A", "BTN_SOUTH": "A",
            "BTN_Y": "Y", "BTN_WEST": "Y",
        }

        self.mode_id = None
        self.estop_flag= False
        self.sleep_flag= False
        self.wake_flag = False
        self.q        = deque(maxlen=128)

        # --- Hold detection (2 sec) for ABS_Z (wake) and ABS_RZ (sleep) ---
        self._hold_sec = 2
        self._abs_z_pressed_since: float | None = None
        self._abs_rz_pressed_since: float | None = None

        self.dev = self._get_dev(timeout_ms=5000)
        self.disconn = self.dev is None

        threading.Thread(target=self._reader, daemon=True).start()

    def _get_dev(self, timeout_ms: int):
        byid_pattern = "/dev/input/by-id/*-event-joystick"
        deadline = time.monotonic() + (timeout_ms / 1000.0)
        while True:
            try:
                candidates = glob.glob(byid_pattern)
                if candidates:
                    dev = InputDevice(candidates[0])
                    return dev
            except Exception:
                pass
            now = time.monotonic()
            if now >= deadline:
                return None
            
            remaining = deadline - now
            time.sleep(0 if remaining < 0.001 else min(0.001, remaining))

    def _reader(self):
        while True:
            if self.disconn or self.dev is None:
                self.dev = self._get_dev(timeout_ms=1000)
                self.disconn = self.dev is None
                time.sleep(0.05)
                continue

            try:
                select.select([self.dev.fd], [], [], 0.1)
            except Exception:
                pass

            batch = []
            try:
                for ev in self.dev.read():   
                    if ev.type == ecodes.EV_ABS:
                        code = ecodes.ABS.get(ev.code, f"ABS_{ev.code}")
                        batch.append((code, ev.value))
                    elif ev.type == ecodes.EV_KEY:
                        code = ecodes.BTN.get(ev.code, ecodes.KEY.get(ev.code, f"KEY_{ev.code}"))
                        batch.append((code, 1 if ev.value else 0))
                        
                if batch:
                    self.q.append(batch)
                    self.disconn = False

            except BlockingIOError:
                pass

            except Exception:
                self.dev = self._get_dev(timeout_ms=1000)
                self.disconn = self.dev is None
                time.sleep(0.05)
                continue

    def _update_mode(self):
        new_mode = None
        hatX, hatY, X, B, A, Y = self.btn_input["hatX"], self.btn_input["hatY"], self.btn_input["X"], self.btn_input["B"], self.btn_input["A"], self.btn_input["Y"]
        if hatY == -1:
            if Y: new_mode = 1
            elif B: new_mode = 2
            elif A: new_mode = 3
            elif X: new_mode = 4
        elif hatX == 1:
            if Y: new_mode = 5
            elif B: new_mode = 6
            elif A: new_mode = 7
            elif X: new_mode = 8
        elif hatY == 1:
            if Y: new_mode = 9
            elif B: new_mode = 10
            elif A: new_mode = 11
            elif X: new_mode = 12
        elif hatX == -1:
            if Y: new_mode = 13
            elif B: new_mode = 14
            elif A: new_mode = 15
            elif X: new_mode = 16
        self.mode_id = new_mode

    def _update_estop_flag(self):
        new_estop_flag = False
        ABS_Z, ABS_RZ = self.btn_input["ABS_Z"], self.btn_input["ABS_RZ"]
        if ABS_Z > 0 and ABS_RZ > 0:
            new_estop_flag = True
        self.estop_flag = new_estop_flag

    # --- 2000ms hold required for sleep (ABS_RZ) ---
    def _update_sleep_flag(self):
        now = time.monotonic()
        rz_active = self.btn_input["ABS_RZ"] > 0

        if rz_active:
            if self._abs_rz_pressed_since is None:
                self._abs_rz_pressed_since = now
            held = (now - self._abs_rz_pressed_since) >= self._hold_sec
            self.sleep_flag = held
        else:
            self._abs_rz_pressed_since = None
            self.sleep_flag = False

    # --- 2000ms hold required for wake (ABS_Z) ---
    def _update_wake_flag(self):
        now = time.monotonic()
        z_active = self.btn_input["ABS_Z"] > 0

        if z_active:
            if self._abs_z_pressed_since is None:
                self._abs_z_pressed_since = now
            held = (now - self._abs_z_pressed_since) >= self._hold_sec
            self.wake_flag = held
        else:
            self._abs_z_pressed_since = None
            self.wake_flag = False

    def get_cmd(self):
        while self.q:
            for code, state in self.q.popleft():
                if isinstance(code, (tuple, list)):
                    std = self.btn_alias.get(code[0])
                    self.btn_input[std] = 1 if state else 0
                    continue

                i = self.idx.get(code, -1)
                if i >= 0:
                    self.stick_input[i] = state
                    continue
                if code == "ABS_HAT0X": self.btn_input["hatX"] = int(state)
                elif code == "ABS_HAT0Y": self.btn_input["hatY"] = int(state)
                elif code == "BTN_TL": self.btn_input["BTN_TL"] = 1 if state else 0
                elif code == "BTN_TR": self.btn_input["BTN_TR"] = 1 if state else 0
                elif code == "ABS_Z":  self.btn_input["ABS_Z"] = int(state)
                elif code == "ABS_RZ": self.btn_input["ABS_RZ"] = int(state)
                else:
                    if isinstance(code, str):
                        std = self.btn_alias.get(code)
                        if std:
                            self.btn_input[std] = 1 if state else 0

        for i in range(4):
            self.robot_cmd[i] = self.stick_input[i]*self.scale[i]
            self.robot_cmd[i] = 0.9 * self.robot_prev_cmd[i] + 0.1 * self.robot_cmd[i]

            if abs(self.robot_cmd[i]) < self.dz_th[i]:
                self.robot_cmd[i] = 0.0 

            if self.robot_cmd[i] > self.max_cmd[i] - 1e-3:
                self.robot_cmd[i] = self.max_cmd[i]
            elif self.robot_cmd[i] < -self.max_cmd[i] + 1e-3:
                self.robot_cmd[i] = -self.max_cmd[i]

            self.robot_prev_cmd[i] = self.robot_cmd[i]


        self.robot_cmd[4] = self.btn_input["BTN_TL"]
        self.robot_cmd[5] = self.btn_input["BTN_TR"]

        self._update_mode()
        self._update_estop_flag()
        self._update_sleep_flag()
        self._update_wake_flag()

        if self.estop_flag:
            raise JoystickEstopError("E-stop triggered by joystick input.")

        if self.sleep_flag:
            raise JoystickAPIError("Sleep triggered by joystick input.")

        return {"cmd_vector": self.robot_cmd, "mode_id": self.mode_id, "estop": self.estop_flag, "wake": self.wake_flag, "sleep": self.sleep_flag}
