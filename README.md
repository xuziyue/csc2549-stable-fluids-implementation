# Stable Fluids Implementation
The Python script stable_fluids.py file contains our implementation of the Stable Fluids simulation algorithm in Stam's paper: Stable Fluids. It is tested with python 3.7/3.8 with the packages in the requirements.txt installed.

## Installation

To run the simulation, first install the required packages.

```bash
pip3 install -r requirements.txt
```

## Usage

We implemented three modes (standard, smoke, picture) and two boundary methods (periodic, fixed).

**Standard interactive mode with fixed boundary**
```bash
python3 stable_fluids.py --fixed_boundary
```

**Standard interactive mode with periodic boundary**
```bash
python3 stable_fluids.py --periodic_boundary
```

**Smoke simulation with fixed boundary.** We don't recommend you to run the smoke simulation with periodic boundary since it is unrealistic in the real world and the simulation is unnatural.
```bash
python3 stable_fluids.py --smoke
```

**Picture interaction mode with fixed boundary.** We don't recommend you to run the picture interaction mode with periodic boundary since it is unrealistic in the real world and the simulation is unnatural.
```bash
python3 stable_fluids.py --picture 
```



Again, You could mix the mode and boundary method to play around, e.g., 'python3 stable_fluids.py --picture --periodic_boundary', 'python3 stable_fluids.py --smoke --periodic_boundary'. However, for both picture mode and smoke mode, we recommend to use the default fixed boundary method, since it is more realistic and natural.

**Reset the simulation**

For all methods, you can press the key 'r' to reset it to original stage at any time.
