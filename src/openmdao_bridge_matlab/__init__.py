import dataclasses
from types import SimpleNamespace
from typing import Any

import numpy as np
import openmdao.api as om

try:
    import matlab.engine
    from matlab.mlarray import _MLArrayMetaClass
except ImportError:
    raise ImportWarning(
        "Matlab Engine API for Python cannot be imported. Install it in the current environment to continue."
    )


matlab_state = SimpleNamespace(engine=None)


def nans(shape):
    return np.ones(shape) * np.nan


DTYPE_MAPPINGS = (
    (bool, matlab.logical),
    (int, matlab.int64),
    (float, matlab.double),
    (np.bool, matlab.logical),
    (np.double, matlab.double),
    (np.single, matlab.single),
    (np.int8, matlab.int8),
    (np.int16, matlab.int16),
    (np.int32, matlab.int32),
    (np.int64, matlab.int64),
    (np.uint8, matlab.uint8),
    (np.uint16, matlab.uint16),
    (np.uint32, matlab.uint32),
    (np.uint64, matlab.uint64),
)

MATLAB_TO_NUMPY_TYPE_MAP = {ml_type: np_type for (np_type, ml_type) in DTYPE_MAPPINGS}


@dataclasses.dataclass(frozen=True)
class MatlabVar:
    name: str
    value: Any
    ml_name: str
    discrete: bool = False
    shape: tuple = (1,)
    units: str = None
    ml_type: _MLArrayMetaClass = None

    def om_to_ml(self, value):
        try:
            # value is a NumPy array
            py_type = value.dtype.type
        except AttributeError:
            py_type = type(value)

        if self.ml_type is not None:
            ml_type = self.ml_type
        elif py_type in (bool, np.bool):
            ml_type = matlab.logical
        else:
            ml_type = matlab.double

        value_arr = np.atleast_2d(value)
        return ml_type(value_arr.tolist(), size=value_arr.shape)

    def ml_to_om(self, value):
        if isinstance(value, _MLArrayMetaClass):
            type_map = MATLAB_TO_NUMPY_TYPE_MAP
            np_type = type_map[type(value)]
            return np.array(value, dtype=np_type).reshape(value.size)
        else:
            return value


class MatlabFunctionComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("working_directory", types=str)
        self.options.declare("function_name", types=str)
        self.options.declare("inputs", types=list)
        self.options.declare("outputs", types=list)
        self.options.declare("desktop", default=False, types=bool)
        self.options.declare("stop_on_error", default=False, types=bool)
        matlab_state.engine = None

    def setup(self):
        for var in self.options["inputs"]:
            if var.discrete:
                self.add_discrete_input(var.name, val=var.value)
            else:
                self.add_input(
                    var.name, val=var.value, shape=var.shape, units=var.units
                )

        for var in self.options["outputs"]:
            if var.discrete:
                self.add_discrete_output(var.name, val=var.value)
            else:
                self.add_output(
                    var.name, val=var.value, shape=var.shape, units=var.units
                )

        self.ensure_matlab_engine()

    def ensure_matlab_engine(self):
        if not matlab_state.engine:
            matlab_state.engine = matlab.engine.connect_matlab()
            matlab_state.engine.eval(
                f"cd(\"{self.options['working_directory']}\");", nargout=0
            )
            if self.options["desktop"]:
                matlab_state.engine.desktop(nargout=0)
            if self.options["stop_on_error"]:
                matlab_state.engine.eval("dbstop if error", nargout=0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        self.ensure_matlab_engine()

        input_data = {}
        for var in self.options["inputs"]:
            if var.discrete:
                inp = discrete_inputs[var.name]
            else:
                inp = inputs[var.name]
            input_data[var.ml_name] = var.om_to_ml(inp)

        output_data = matlab_state.engine.feval(
            self.options["function_name"],
            *input_data.values(),
            nargout=len(self.options["outputs"]),
        )

        for var_idx, var in enumerate(self.options["outputs"]):
            # outp = output_data[var_map.ext_name]
            outp = var.ml_to_om(output_data[var_idx])
            if var.discrete:
                discrete_outputs[var.name] = outp
            else:
                outputs[var.name] = outp

    def cleanup(self):
        if matlab_state.engine:
            matlab_state.engine.quit()
        matlab_state.engine = None
        super().cleanup()
