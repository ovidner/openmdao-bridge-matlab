import hypothesis.extra.numpy as np_st
import numpy as np
import openmdao.api as om
from hypothesis import given, settings
from openmdao_bridge_matlab import MatlabFunctionComp, MatlabVar


@given(np_st.arrays(np.float, np_st.array_shapes()))
@settings(max_examples=10, deadline=30000)
def test_continuous(value):
    prob = om.Problem()
    model = prob.model

    model.add_subsystem("indeps", om.IndepVarComp("x", val=value))
    model.add_subsystem(
        "passthrough",
        MatlabFunctionComp(
            function_name="tests/data/passthrough.m",
            inputs=[MatlabVar(name="in", value=np.nan, ml_name="a", shape=value.shape)],
            outputs=[
                MatlabVar(name="out", value=np.nan, ml_name="b", shape=value.shape)
            ],
        ),
    )
    model.connect("indeps.x", "passthrough.in")

    try:
        prob.setup()
        prob.run_model()
    finally:
        prob.cleanup()

    # Using a normal == comparison will not consider NaNs as equal.
    assert np.allclose(prob["indeps.x"], value, atol=0.0, rtol=0.0, equal_nan=True)
    assert np.allclose(
        prob["passthrough.out"], prob["indeps.x"], atol=0.0, rtol=0.0, equal_nan=True
    )
