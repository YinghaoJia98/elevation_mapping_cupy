import cupy as cp
import string

def frontier_kernel(
    StepMinimumValue
):
    frontier_kernel = cp.ElementwiseKernel(
        in_params="raw T h, raw T tra",
        out_params="T h_frontier",
        operation=string.Template(
            """
            int row = i / h.shape()[1];  // row index of current element
            int col = i % h.shape()[1];  // column index of current element
            double StepMinimumValue=${StepMinimumValue};
            double height = h[i];
            double tra_score = tra[i];
            if( (height!=height) || (tra_score!=tra_score) || (tra_score<StepMinimumValue) )
            {
                h_frontier = 0;
            }
            else
            {
                if( (row == (h.shape()[0] - 1)) || (col == (h.shape()[1] - 1)) || (row == 0) || (col == 0) )
                {
                    //printf("hello/n");
                    h_frontier = 1;
                }
                else
                {
                    int ValidCount=0;
                    for (int r = -1; r <= 1; r++)
                    {
                        for (int c = -1; c <= 1; c++) 
                        {
                            int nr = row + r;  // row index of neighbor
                            int nc = col + c;  // column index of neighbor
                            if (nr >= 0 && nr < h.shape()[0] && nc >= 0 && nc < h.shape()[1]) 
                            {
                                if( (h[nr * h.shape()[1] + nc] == h[nr * h.shape()[1] + nc]) && (tra[nr * h.shape()[1] + nc] == tra[nr * h.shape()[1] + nc]) )
                                {
                                    ValidCount++;
                                }   
                            }
                        }
                    }
                    if(ValidCount > 8)
                    {
                        h_frontier = 0;
                    }
                    else
                    {
                        h_frontier = 1;
                    }
                }
            }
            """
        ).substitute(
            StepMinimumValue=StepMinimumValue,
        ),
        name="frontier_kernel",
    )
    return frontier_kernel