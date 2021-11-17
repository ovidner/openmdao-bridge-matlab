# OpenMDAO-Bridge-MATLAB
[![DOI](https://zenodo.org/badge/DOI/10/gpqk.svg)](https://doi.org/gpqk)

An OpenMDAO component for running analyses in MATLAB.

## Prerequisites/Installation

You will need a Conda environment to install the bridge in. Within your Conda
environment, run the following:

    conda install -c ovidner openmdao-bridge-matlab

## Usage

Let's say you have a MATLAB function defined like so:

```matlab
function [out_a, out_b] = fun(in_a, in_b)

out_a = ...
out_b = ...

end
```

Then you can use it as an OpenMDAO component, perhaps something like this:

```python
import openmdao.api as om
from openmdao_bridge_matlab import MatlabFunctionComp, MatlabVar

prob = om.Problem()

matlab_comp = MatlabFunctionComp(
    function_name="fun",
    inputs=[
        MatlabVar(name="in_1", ml_name="in_a", value=1),
        MatlabVar(name="in_2", ml_name="in_b", value=1),
    ],
    outputs=[
        MatlabVar(name="out_1", ml_name="out_a", value=0, units="m"),
        MatlabVar(name="out_2", ml_name="out_b", value=0),
    ],
)
prob.model.add_subsystem("matlab_comp", matlab_comp)

prob.setup()
prob.set_val("matlab_comp.in_1", 5)
prob.run_model()
prob.get_val("matlab_comp.out_1", units="mm")
```
