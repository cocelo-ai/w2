class RobotEStopError(Exception):
    """Exception raised when the robot's emergency stop is activated."""
    def __init__(self, message=""):
        self.message = message
        super().__init__(self.message)

class RobotInitError(Exception):
    """Exception raised when the robot's emergency stop is activated."""
    def __init__(self, message=""):
        self.message = message
        super().__init__(self.message)

class RobotAPIError(Exception):
    def __init__(self, message=""):
        self.message = message
        super().__init__(self.message)

class RobotSetGainsError(Exception):
    """Exception raised when setting the robot's gains fails"""
    def __init__(self, message=""):
        self.message = message
        super().__init__(self.message)

class ControlRateError(Exception):
    """Exception raised for errors in the control rate decorator."""
    def __init__(self, message=""):
        self.message = message
        super().__init__(self.message)

class ModeConfigError(Exception):
    """Exception raised for errors in the mode configuration."""
    def __init__(self, message=""):
        self.message = message
        super().__init__(self.message)  