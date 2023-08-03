import cupy as cp
import string


def normalize_kernel(
    StepMinimumValue
):
    normalize_kernel = cp.ElementwiseKernel(
        in_params="raw T h, raw T step",
        out_params="T h_normalize",
        operation=string.Template(
            """
            double StepMinimumValue=${StepMinimumValue};
            int row = i / h.shape()[1];  // row index of current element
            int col = i % h.shape()[1];  // column index of current element
            double height = h[i];
            double step_score = step[i];
            if((height!=height) || (step_score!=step_score) || (step_score<StepMinimumValue))
            {
            h_normalize=0;
            }
            else
            {
            h_normalize=1;
            }
            """
        ).substitute(
            StepMinimumValue=StepMinimumValue,
        ),
        name="normalize_kernel",
    )
    return normalize_kernel