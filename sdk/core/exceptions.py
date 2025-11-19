class JoystickEstopError(Exception):
    def __init__(self, message=""):
        self.message = message
        super().__init__(self.message)

class JoystickAPIError(Exception):
    def __init__(self, message=""):
        self.message = message
        super().__init__(self.message)
