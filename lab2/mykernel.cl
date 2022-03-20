//
//  Based on the formula for Pi, compute the equation :
//    SUM{ [(1/4n+1)-(1/4n+3)] } (n = 0,1,2,3....)
//

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void Picalculation(
    __global float *pi_out,
    int pair_per_item)
{
    int c = 4;
    
    // Define denominator of the equation
    int i,d;
    
    // Initialize the numerator and equation result
    float n = 1.0f;
    float e = 0.0f;
    
    /* Make sure previous processing has completed. All work-items in a work-group executing the kernel on a processor must execute this function before any are allowed to continue execution beyond the barrier. */
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Get global work-item id.
    int work_item = get_global_id (0);

    // clear the buffer
    pi_out[work_item] = 0;

    // Offset
    int index = work_item * pair_per_item * c;
    
    // The equation for work-item. Iteration calculate all pairs.
    for (i = 0; i < pair_per_item; i++)
    {
        d = i * c + index;
        e = (n/(d + 1)) - (n/(d + 3));
        pi_out[work_item] += e;
    }

}


