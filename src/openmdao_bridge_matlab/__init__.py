import dataclasses
import pathlib
from typing import Any

import numpy as np
import openmdao.api as om
import pymatbridge as pmb


def nans(shape):
    return np.ones(shape) * np.nan


@dataclasses.dataclass(frozen=True)
class MatlabVar:
    name: str
    value: Any
    ml_name: str = ""
    discrete: bool = False
    shape: tuple = (1,)
    units: str = None
    # TODO: this is ugly. come up with an elegant way to solve it!
    semvar: object = None


class MatlabFunctionComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("function_name", types=(str, pathlib.Path))
        self.options.declare("inputs", types=list)
        self.options.declare("outputs", types=list)
        self.options.declare("desktop", default=False, types=bool)
        self.options.declare("stop_on_error", default=False, types=bool)
        self.options.declare("session", types=pmb.Matlab, default=pmb.Matlab())
        self.options.declare("startup_options", types=str, default="default")
        self.options.declare(
            "preheat",
            types=bool,
            default=True,
            desc="Start the MATLAB session as part of setup. If False, it will start on first computation.",
        )

    def setup(self):
        for var in self.options["inputs"]:
            if var.semvar:
                var.semvar.add_as_input(self)
                continue

            if var.discrete:
                self.add_discrete_input(var.name, val=var.value)
            else:
                self.add_input(
                    var.name, val=var.value, shape=var.shape, units=var.units
                )

        for var in self.options["outputs"]:
            if var.semvar:
                var.semvar.add_as_output(self)
                continue

            if var.discrete:
                self.add_discrete_output(var.name, val=var.value)
            else:
                self.add_output(
                    var.name, val=var.value, shape=var.shape, units=var.units
                )

        if self.options["preheat"] and not self.options["session"].started:
            self.options["session"].start()

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        session = self.options["session"]
        input_data = {}
        for var in self.options["inputs"]:
            if var.discrete:
                inp = discrete_inputs[var.name]
            else:
                inp = inputs[var.name]
            input_data[var.ml_name] = inp

        num_outs = len(self.options["outputs"])

        if not session.started:
            session.start()

        response = session.run_func(
            self.options["function_name"], *input_data.values(), nargout=num_outs,
        )

        if not response["success"]:
            raise om.AnalysisError(
                f'MATLAB session "{session}" failed with message: {response["content"]["stdout"]}'
            )

        result = response["result"]

        for var_idx, var in enumerate(self.options["outputs"]):
            # output_data is a scalar if only one output
            outp = result if num_outs == 1 else result[var_idx]
            if var.discrete:
                discrete_outputs[var.name] = outp
            else:
                outputs[var.name] = outp

    def cleanup(self):
        self.options["session"].stop()
        super().cleanup()
