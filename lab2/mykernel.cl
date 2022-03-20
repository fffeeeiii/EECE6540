/*
 Based on the Leibniz formula for Pi, the equation is:
 Pi/4 = sigma summation [(1/4n+1)-(1/4n+3)] goes from 0 to infinity
 */
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void Picalculation(
    __global float *pi_out)
{
    // Define the fractions amount on each work-item
    int fraction_per_item = 16 / 2;
    int c = 4;
    
    // Define denominator of Leibniz equation
    int i,d;
    
    // Initialize the numerator and equation result
    float n = 1.0f;
    float e = 0.0f;
    
    /* Make sure previous processing has completed. All work-items in a work-group executing the kernel on a processor must execute this function before any are allowed to continue execution beyond the barrier. */
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Get global work-item id.
    int work_item = get_global_id (0);
    
    // Offset
    int term = work_item * fraction_per_item * c;
    
    // The equation for work-item. Iteration calculate all fractions.
        for (i = 0; i < fraction_per_item; i++)
        {
            d = i * c + term;
            e = (n/(d + 1)) - (n/(d + 3));
            pi_out[work_item] += e;
        }
 }


