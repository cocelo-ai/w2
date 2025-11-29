import time
from functools import wraps

from sdk import *


def routine_control_rate():
    period_ns = 20_000_000   # 20 ms (50 Hz)
    busy_spin_ns = 400_000
    wake_ahead_ns = 500_000

    def decorator(loop_func):
        @wraps(loop_func)
        def runner(*args, **kwargs):
            try:
                start_call_ns = time.monotonic_ns()
                next_tick = start_call_ns + period_ns
                while True:
                    loop_func(*args, **kwargs)

                    now = time.monotonic_ns()
                    remaining = next_tick - now
       
                    if remaining <= 0:
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
                return
            except Exception as e:
                robot.estop()
                return
            except BaseException as e:
                robot.estop()
                return
        return runner
    return decorator


@routine_control_rate()
def wake(robot: Robot):
    pass