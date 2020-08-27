import numpy as np
import openmdao.api as om

try:
    import matlab.engine
except ImportError:
    raise ImportWarning(
        "Matlab Engine API for Python cannot be imported. Install it in the current environment to continue."
    )


def nans(shape):
    return np.ones(shape) * np.nan


class MatlabScriptComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("script_path", types=str)
        self.options.declare("inputs", types=list)
        self.options.declare("outputs", types=list)
        self.engine = None

    def setup(self):
        for var_map in self.options["inputs"]:
            self.add_input(name=var_map.name, val=nans(var_map.shape))

        for var_map in self.options["outputs"]:
            self.add_output(name=var_map.name, val=nans(var_map.shape))

        self.engine = matlab.engine.start_matlab()

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        self.engine.eval("clear all;", nargout=0)

        for var_map in self.options["inputs"]:
            self.engine.workspace[var_map.ext_name] = matlab.double(
                inputs[var_map.name].tolist()
            )

        self.engine.run(self.options["script_path"], nargout=0)

        for var_map in self.options["outputs"]:
            outputs[var_map.name] = np.array(
                self.engine.workspace[var_map.ext_name]
            ).reshape(var_map.shape)

    def cleanup(self):
        self.engine.quit()
        self.engine = None
        super().cleanup()
