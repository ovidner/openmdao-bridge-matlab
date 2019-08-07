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


class MatlabScriptComponent(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("script_path", types=str)
        self.options.declare("inputs", types=dict)
        self.options.declare("outputs", types=dict)

    def setup(self):
        for name, opts in self.options["inputs"].items():
            self.add_input(name=name, val=nans(opts["shape"]))

        for name, opts in self.options["outputs"].items():
            self.add_output(name=name, val=nans(opts["shape"]))

        self.engine = matlab.engine.start_matlab()

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        self.engine.eval("clear all;", nargout=0)

        for name, opts in self.options["inputs"].items():
            self.engine.workspace[opts["matlab_name"]] = inputs[name].tolist()

        self.engine.run(self.options["script_path"], nargout=0)

        for name, opts in self.options["outputs"].items():
            outputs[name] = np.array(self.engine.workspace[opts["matlab_name"]])

    def cleanup(self):
        self.engine.quit()
        super().cleanup()
