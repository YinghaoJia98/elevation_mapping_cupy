import cupy as cp
import string


def cutter_min_kernel(
    MinimumValue,
    first_num
):
    cutter_min_kernel = cp.ElementwiseKernel(
        in_params="raw T h",
        out_params="T h_curved",
        operation=string.Template(
            """
            int first_num=${first_num};
            double MinimumValue=${MinimumValue};
            int row = i / h.shape()[1];  // row index of current element
            int col = i % h.shape()[1];  // column index of current element
            double height = h[i];
            double heightSum = 0.0;
            double heightOld = h[i];
            int Count=0;
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
                        if(height!=height)
			            {
			                //height=0;
			                //printf("hello");
			                continue;
			            }
                        if( (r == 0 ) && ( c == 0 ) )
                        {
                            continue;
                        }
                        heightSum = heightSum + height;
                        Count = Count + 1;
                    }
                }
            }
            if(Count>0)
            {
                double heightMean = heightSum / Count;
                if(( heightOld < (heightMean - MinimumValue)) && (Count > (0.95 * 4 * first_num * first_num)))
                {
                    h_curved = heightMean;
                }
                else
                {
                    h_curved = heightOld;
                }
            }  
            """
        ).substitute(
            MinimumValue=MinimumValue,
            first_num=first_num,
        ),
        name="cutter_min_kernel",
    )
    return cutter_min_kernel