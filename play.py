from sdk import *

#Robot's Gains
kp = [70, 70, 70, 70, 70, 70, 0, 0]
kd = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.55, 0.55]

# RL mode
mode = Mode(mode_cfg={
    "id" : 1,
    "stacked_obs_order": ["dof_pos", "dof_vel", "ang_vel", "proj_grav", "last_action"],
    "non_stacked_obs_order": ["command"],
    "obs_scale": {"dof_vel": 0.15,
                  "ang_vel": 0.25,
                  "command": [2.0, 0.0, 0.25, 0.0],
                  },
    "action_scale": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 40.0, 40.0],
    "stack_size": 3,
    "policy_path": "weight/policy.onnx",
    "cmd_vector_length": 4,
})


# Instances
robot = Robot()
joystick = Joystick(max_cmd=[1, 0, 1, 0])
rl = RL()

# Set gains
robot.set_gains(kp=kp, kd=kd)

# Add & Set Mode
rl.add_mode(mode)
rl.set_mode(mode_id=1)

# Wake the robot
#wake(robot)

@control_rate(robot, hz=50)
def loop():
    obs = robot.get_obs()             # Get observation
    cmd = joystick.get_cmd()          # Get command
    state = rl.build_state(obs, cmd)  # Build state
    action = rl.select_action(state)  # Select action
    robot.do_action(action)
    
loop()
