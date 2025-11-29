import time
from functools import wraps
import traceback
import gc

from sdk import *
from ..classes.logger import Logger

def control_rate(robot: Robot,
                hz: float = 50.0,
                busy_spin_ns: int = 400000,
                wake_ahead_ns: int = 500):
    gc.disable()
    logger=Logger()

    hz = max(1e-5, hz)
    busy_spin_ns = max(0, busy_spin_ns)
	
    period_ns = int(1000000000 / float(hz))
    wake_ahead_ns = max(0, min(wake_ahead_ns, busy_spin_ns))

    def decorator(loop_func):
        @wraps(loop_func)
        def runner(*args, **kwargs):
            try:
                cnt = 0
                start_call_ns = time.monotonic_ns()
                next_tick = start_call_ns + period_ns
                while True:
                    cnt += 1
 
                    loop_func(*args, **kwargs)

                    now = time.monotonic_ns()
                    remaining = next_tick - now

                    print(f"\nElapsed: {(period_ns - remaining) / 1000000} ms\n")
                    
                    if remaining <= 0:
                        logger.warning(f"[cnt= {cnt}] Control loop overrun: {-remaining / 1000000:.6f} ms")
                        next_tick = time.monotonic_ns() + period_ns
                        continue

                    sleep_ns = remaining - busy_spin_ns
                    if sleep_ns > 0:
                        time.sleep(sleep_ns / 1000000000)

                    while time.monotonic_ns() < (next_tick - wake_ahead_ns):
                        pass

                    next_tick += period_ns

            except KeyboardInterrupt as e:
                logger.critical("Control loop interrupted by user (KeyboardInterrupt).")
                robot.do_action([0] * 8, torque_ctrl=True)
                #robot.estop()  # TODO: Activate estop()
                return
            except JoystickEstopError as e:
                logger.critical(f"{e}")
                robot.estop()
                return
            except JoystickSleepError as e:
                logger.critical(f"{e}")
                #sleep(robot) # TODO: Make Sleep Mode
                return
            except RobotEStopError as e:
                logger.critical(f"{e}")
                return
            except Exception as e:
                logger.critical(f"{e}\n{traceback.format_exc()}")
                robot.estop()
                return
            except BaseException as e:
                logger.critical(f"{e}\n{traceback.format_exc()}")
                robot.estop()
                return
        return runner
    return decorator
