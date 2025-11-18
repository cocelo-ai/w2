#!/usr/bin/env bash

# --- Pick Python with conda priority ---
if [[ -n "${PYTHON_EXEC:-}" ]]; then
  PYTHON="${PYTHON_EXEC}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
else
  PYTHON="$(command -v python3)"
fi

# ===== Smoke test =====
echo "== Smoke test =="
"${PYTHON}" - <<'PY'
import os, random, sys
print("Python:", sys.executable)

# --- Smoke test (onnxpolicy) ---------------------------------------------
from sdk import *

pol = MLPPolicy("weight/policy.onnx") # no_arg test
obs = [random.uniform(-0.5, 0.5) for _ in range(88)]

action = pol.inference(obs)   # mlp policy inference test
print("action:", action)

print("=========================================================")
print("============    import onnxpolicy ... OK!    ============")
print("=========================================================\n")

# --- Smoke test (mode) ---------------------------------------------
mode = Mode(mode_cfg={
    "id" : 1,
    "stacked_obs_order": ["dof_pos", "dof_vel", "ang_vel", "proj_grav", "last_action"],
    "non_stacked_obs_order": ["command"],
    "obs_scale": {"dof_vel": 0.5,
                  "ang_vel": 1,
                  "command": [2, 1.0, 0.25, 1.0]},
    "action_scale": [10.0]*8,
    "stack_size": 3,
    "policy_path": "weight/policy.onnx",
    "cmd_vector_length": 4,
})
import numpy as np
obs = [random.uniform(-0.5, 0.5) for _ in range(88)]
action = mode.inference(obs)  # only accept  1D list/array
print("len(action):", len(action))
print("type(action)", type(action))

print("=========================================================")
print("===============    import mode ... OK!    ===============")
print("=========================================================\n")

rl = RL()
rl.add_mode(mode)
rl.set_mode(mode_id=1)
obs1 = {
        "dof_pos": [1, 2, 3, 4, 5, 6],
        "dof_vel": [10, 20, 30, 40, 50, 60, 70, 80],
        "ang_vel": [100, 200, 300],
        "proj_grav": [1000, 2000, 3000],
}
obs2 = {
        "dof_pos": [1.0] * 6,
        "dof_vel": [1.0] * 8,
        "ang_vel": [1.0] * 3,
        "proj_grav": [1.0] * 3,
}
obs3 = {
        "dof_pos": [2.0] * 6,
        "dof_vel": [2.0] * 8,
        "ang_vel": [2.0] * 3,
        "proj_grav": [2.0] * 3,
}

obs4 = {
        "dof_pos": [3.0] * 6,
        "dof_vel": [3.0] * 8,
        "ang_vel": [3.0] * 3,
}

obs5 = {}

cmd = {"cmd_vector": [0.7, 0.7, 0.7, 0.7]}
state1 = rl.build_state(obs1, cmd)
action = rl.select_action(state1)
print("action:", action)
state2 = rl.build_state(obs2, cmd)
action = rl.select_action(state2)
print("action:", action)
state3 = rl.build_state(obs3, cmd)
action = rl.select_action(state3)
print("action:", action)

t=[]
t.extend(state3[:28])
t.extend(state2[:28])
t.extend(state1[:28])
scaled_cmd_vector = [cmd["cmd_vector"][0]*2.0, cmd["cmd_vector"][1]*1.0, cmd["cmd_vector"][2]*0.25, cmd["cmd_vector"][3]*1.0]
t.extend(scaled_cmd_vector)


state4 = rl.build_state(obs4, cmd)
action = rl.select_action(state4)
state4 = rl.build_state(obs5, cmd, last_action = [9] * 8)

print("state1", state1, "\n\n")
print("state2", state2, "\n\n")
print("state3", state3, "\n\n")


diff = []
for i in range(88):
    diff.append(t[i] - state3[i])
print("<<<< TOTAL STATE DIFF! >>>>\n", diff, '\n')

print("external_last_action: ", state4, "\n\n")
print("=========================================================")
print("===============    import rl ... OK!    ===============")
print("=========================================================\n")

rl = RL()
mode = Mode(mode_cfg={
    "id" : 1,
    "stacked_obs_order": ["dof_pos", "dof_vel", "ang_vel", "proj_grav", "last_action"],
    "non_stacked_obs_order": ["command"],
    "obs_scale": {"dof_vel": 1,
                  "ang_vel": 1,
                  "command": 1},
    "action_scale": 1,
    "stack_size": 3,
    "policy_path": "weight/policy.onnx",
    "cmd_vector_length": 4,
})
rl.add_mode(mode)
rl.set_mode(mode_id=1)
robot = Robot()
cmd = {"cmd_vector": [0.7, 0.7, 0.7, 0.7]}
obs = robot.get_obs()
state = rl.build_state(obs, cmd)
action = rl.select_action(state)

onnxpolicy = MLPPolicy("weight/policy.onnx")
onnx_action = onnxpolicy.inference(state)

print("onnx_action", onnx_action)
print("action:", action, "\n")
print("obs:", obs, "\n")
print("state", state, "\n")

obs = robot.get_obs()
state = rl.build_state(obs, cmd)
action = rl.select_action(state)
print("action:", action, "\n")
print("obs:", obs, "\n")
print("state", state, "\n")

print("=========================================================")
print("===============    import robot ... OK!    ===============")
print("=========================================================\n")
PY
