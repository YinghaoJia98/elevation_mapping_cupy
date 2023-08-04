import cupy as cp
import string


def normalize_kernel(
):
    normalize_kernel = cp.ElementwiseKernel(
        in_params="raw T h, raw T step",
        out_params="T h_normalize",
        operation=string.Template(
            """
            int row = i / h.shape()[1];  // row index of current element
            int col = i % h.shape()[1];  // column index of current element
            double height = h[i];
            double step_score = step[i];
            if((height!=height) || (step_score!=step_score))
            {
                h_normalize=0;
            // h_normalize = nanf("");
            }
            else
            {
                h_normalize=1;
            }
            """
        ).substitute(
        ),
        name="normalize_kernel",
    )
    return normalize_kernel

def merge_traversability_kernel(
    StepMinimumValue
):
    merge_traversability_kernel = cp.ElementwiseKernel(
        in_params="raw T h, raw T h_probability, raw T step",
        out_params="T h_merged",
        operation=string.Template(
            """
            double StepMinimumValue=${StepMinimumValue};
            double height = h[i];
            double probability = h_probability[i];
            double step_score = step[i];
            if( (height!=height) || (step_score<StepMinimumValue) || (step_score!=step_score) )
            {
                h_merged=0;
            }
            else
            {
                h_merged = probability * step_score;
            }
            """
        ).substitute(
            StepMinimumValue=StepMinimumValue,
        ),
        name="merge_traversability_kernel",
    )
    return merge_traversability_kernel