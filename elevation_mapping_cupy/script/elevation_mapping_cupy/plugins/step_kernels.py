import cupy as cp
import string


def compute_hstep_kernel(
    first_num,
):
    compute_hstep_kernel = cp.ElementwiseKernel(
        in_params="raw T h",
        out_params="T h_step",
        operation=string.Template(
            """
            int first_num=${first_num};
            int row = i / h.shape()[1];  // row index of current element
            int col = i % h.shape()[1];  // column index of current element
            double height = h[i];
            bool init = false;
            double heightMax = 0.0;
            double heightMin = 0.0;
            // loop through the 3x3 neighborhood centered at (row, col)
            for (int r = -1*first_num; r <= first_num; r++)
            {
                for (int c = -1*first_num; c <= first_num; c++) 
                {
                    int nr = row + r;  // row index of neighbor
                    int nc = col + c;  // column index of neighbor
                    if (nr >= 0 && nr < h.shape()[0] && nc >= 0 && nc < h.shape()[1]) 
                    {
                        height=h[nr * h.shape()[1] + nc];
                        if(!init)
                        {
                            heightMax=height;
                            heightMin=height;
                            init=true;
                        }
                        else
                        {
                            if(height>heightMax)
                            {
                                heightMax=height;
                            }
                            if(height<heightMin)
                            {
                                heightMin=height;
                            }
                        }
                    }
                }
            }
            if(init)
            {
                double step_height=heightMax-heightMin;
                h_step=step_height;
            }  
            """
        ).substitute(
            first_num=first_num,
        ),
        name="compute_hstep_kernel",
    )
    return compute_hstep_kernel


def compute_hscore_kernel(
    second_num,
    critical_value,
    critical_cell_num,
):
    compute_hscore_kernel = cp.ElementwiseKernel(
        in_params="raw T h_step",
        out_params="T h_score",
        operation=string.Template(
            """
            int second_num=${second_num};
            double critical_value=${critical_value};
            double critical_cell_num_double=(double)${critical_cell_num};
            int row = i / h_step.shape()[1];  // row index of current element
            int col = i % h_step.shape()[1];  // column index of current element
            int nCells=0;
            double stepMax=0.00;
            bool isValid=false;
            
            // loop through the 3x3 neighborhood centered at (row, col)
            for (int r = -1*second_num; r <= second_num; r++)
            {
                for (int c = -1*second_num; c <= second_num; c++) 
                {
                    int nr = row + r;  // row index of neighbor
                    int nc = col + c;  // column index of neighbor
                    if (nr >= 0 && nr < h_step.shape()[0] && nc >= 0 && nc < h_step.shape()[1]) 
                    {
                        isValid=true;
                        if(h_step[nr * h_step.shape()[1] + nc]>stepMax)
                        {
                            stepMax=h_step[nr * h_step.shape()[1] + nc];
                        }
                        if(h_step[nr * h_step.shape()[1] + nc]>critical_value)
                        {
                            nCells++;
                        }
                    }
                }
            }
            if(isValid)
            {
                double nCells_double = (double)nCells;
                double step_middle=nCells_double/critical_cell_num_double*stepMax;
                double step;
                if(stepMax<step_middle)
                {
                    step=stepMax;
                }
                else
                {
                    step=step_middle;
                }
                //double step=std::min(stepMax,step_middle);
                if(step<critical_value)
                {
                    h_score=1.0-step/critical_value;
                }
                else
                {
                    h_score=0.0;
                }
            }  
            """
        ).substitute(
            second_num=second_num,
            critical_value=critical_value,
            critical_cell_num=critical_cell_num,
        ),
        name="compute_hscore_kernel",
    )
    return compute_hscore_kernel
