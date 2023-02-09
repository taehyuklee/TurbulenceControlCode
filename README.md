# TurbulenceControlCode
#### Control of turublent flow with Deep Reinforcement Learning (Test Code)
<span style="color:red"> 글씨색 변경 </span>
Welcome to my repository

This repository contains the codes mentioned in the paper <span style="color:red"> "Turbulence Control for Drag Reduction through deep reinforcement learning" </span> (https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.8.024604?ft).


This repository has three directories (packages) containing three DRL model files.


DoubleQdu - only streamwise wall shear stress field data is used to train the model.

DoubleQdudw - Both streamwise and spanwise wall shear stress field data are utilized to train the model.

DoubleQdw - only spanwise wall shear stress field data is used to train the model.



Each package contains the following Python scripts.
1. Environment.py (Fluid simulator connected to DRL model)
2. actor net.py (actor network)
3. critic net.py (critic network)
4. main xx.py (main DRL algorithm)
5. monitoring.py (it is for monitoring of fluid behavior in learning and statistics for turbulence)
6. ou noise.py (we formerly utilized the ou noise.py script, but we no longer do so you do not need to care about this file)

in addition, we used version 1 of TensorFlow

-----------------------------------------------------------------------------
Additionally, Each model package has the TestCode directory.

This directory provides meta data for trained weight and bias, as well as Test code.

Test.py provides environment and actor model configuration code for the drag reduction test.

And other Environment.py, actor.py, and critic.py are identical to files in the upper directory.


thank you
