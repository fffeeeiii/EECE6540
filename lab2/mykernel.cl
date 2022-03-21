//
//  Based on the formula for Pi, compute the equation :
//    SUM{ [(1/4n+1)-(1/4n+3)] } (n = 0,1,2,3....)
//

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void Picalculation(
    __global double *buffer_p,
    int pair_per_item,
    int local_size,
    __global double *result_p)
{
    int c = 4;
    
    // Define the index for the denominator
    int i,d;
    
    // Initialize the numerator and equation result
    double n = 1.0f;
    double e = 0.0f;
    
    // Make sure previous processing has completed 
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    // Get global ID, group ID and local ID.
    int global_id = get_global_id(0);
    int group_id = get_group_id(0);
    int local_id = get_local_id(0);

    // clear the buffer
    buffer_p[global_id] = 0;    

    // Offset
    int index = global_id * pair_per_item * c;
    
    // The equation for work-item. Iteration calculate all pairs.
    for (i = 0; i < pair_per_item; i++)
    {
        d = i * c + index;
        e = (n/(d + 1)) - (n/(d + 3));
        buffer_p[global_id] += e;
    }
    
    // Make sure local processing has completed */
    barrier(CLK_GLOBAL_MEM_FENCE);
    if(local_id == 0) {
        result_p[group_id] = 0;
        for (i = global_id; i < (global_id + local_size); i++)
        {
            result_p[group_id] += buffer_p[i];
        }
    }

}


