import hypothesis.extra.numpy as np_st
import hypothesis.strategies as st
import numpy as np
import openmdao.api as om
from hypothesis import given, settings

from openmdao_bridge_matlab import MatlabScriptComponent


@given(np_st.arrays(np.float, np_st.array_shapes()))
@settings(max_examples=10, deadline=10000)
def test_continuous(value):
    prob = om.Problem()
    model = prob.model

    model.add_subsystem("indeps", om.IndepVarComp("x", val=value))
    model.add_subsystem(
        "passthrough",
        MatlabScriptComponent(
            script_path="tests/data/passthrough.m",
            inputs={"in": {"matlab_name": "a", "shape": value.shape}},
            outputs={"out": {"matlab_name": "b", "shape": value.shape}},
        ),
    )
    model.connect("indeps.x", "passthrough.in")

    prob.setup()
    prob.run_model()
    prob.cleanup()

    # Using a normal == comparison will not consider NaNs as equal.
    assert np.allclose(prob["indeps.x"], value, atol=0.0, rtol=0.0, equal_nan=True)
    assert np.allclose(
        prob["passthrough.out"], prob["indeps.x"], atol=0.0, rtol=0.0, equal_nan=True
    )
