from sdk import *

# Robot's Gains
#kp = [70, 70, 70, 70, 70, 70, 0, 0]
#kd = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.55, 0.55]

div_coef = 16

kp = [70/div_coef, 70/div_coef, 70/div_coef, 70/div_coef, 70/div_coef, 70/div_coef, 0, 0]
kd = [0.7/div_coef, 0.7/div_coef, 0.7/div_coef, 0.7/div_coef, 0.7/div_coef, 0.7/div_coef, 0.55/div_coef, 0.55/div_coef]

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
joystick = Joystick(max_cmd=[1.0, 0, 1.0, 0])
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
    #robot.do_action([0]*8, torque_ctrl=True)
    
    '''
    dof_pos = obs["dof_pos"]
    dof_vel = obs["dof_vel"]
    kp_vec, kd_vec = robot.get_gains()

    tau = [0.0] * 8

    # ----- 0~5: position target -----
    # action[i] = q_des (관절 목표 각도, rad)
    for i in range(6):
        q_cur = dof_pos[i]
        qd_cur = dof_vel[i]
        q_des = action[i]        # 이미 [-0.5,0.5] 범위로 스케일된 값이라고 가정

        pos_err = q_des - q_cur
        # 표준 PD: tau = Kp * (q_des - q) - Kd * qd
        tau[i] = kp_vec[i] * pos_err - kd_vec[i] * qd_cur

    # ----- 6~7: wheel velocity target -----
    # action[i] = v_des (바퀴 목표 속도, rad/s)
    for i in range(6, 8):
        v_cur = dof_vel[i]
        v_des = action[i]        # [-40, 40] rad/s 범위
        vel_err = v_des - v_cur

        # 단순 속도 제어: tau = Kd * (v_des - v_cur)
        # (kp[6], kp[7] = 0 이라 purely D control)
        tau[i] = kd_vec[i] * vel_err

    robot.do_action(tau, torque_ctrl=True)           # Do action
    '''
    
loop()
