# Cloth Manipulation Simulation Environment

This repository contains a physics-based simulation environment for **cloth manipulation tasks**, designed for research in robot learning, imitation learning, and sim-to-real evaluation. 

The environment supports contact-rich interactions between deformable cloth and robot URDFs, and is intended to be used for demo collection, policy training, and evaluation. It is designed to support the [2026 WBCD competition](https://wbcdcompetition.github.io/). Please refer to the details of the deformable manipulation task in the competition [here](https://wbcdcompetition.github.io/competition-tracks.html#dm).

The simulator is based on [NVIDIA Newton](https://github.com/newton-physics/newton). Although Newton has flexible support for deformable objects like clothes, it is still in **active beta development** stage. Thus, this repo is still subject to updates and needs case-specific integration into existing robot learning frameworks. Please be aware when adopting this environment for your own use.

---

## Installation

```
# clone the repo
git clone --recurse-submodules git@github.com:kywind/cloth_sim.git
cd cloth_sim

# create and activate a python venv
uv venv --python=3.11
source .venv/bin/activate

# install newton
cd newton
uv pip install -r pyproject.toml --extra examples

# (optionally) verify installation by running newton examples
python -m newton.examples robot_h1

# install main dependencies
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# additional packages
uv pip install opencv-python omegaconf hydra-core pynput transforms3d ipdb joycon-python hid pyglm loop-rate-limiters mink
```

---

## Usage

```
python experiments/demo.py
```

This command runs a manually defined robot action trajectory, shown as follows:

https://github.com/user-attachments/assets/c8cdb991-998f-4cec-9853-c7a65b4a8f7d
