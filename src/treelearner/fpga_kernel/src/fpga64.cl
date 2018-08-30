//kernel for FPGA

#define NUM_BINS 64

//__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1)))
__kernel void kernel_fpgahistogram(
    //                                                       arg# write# read#
             const int num_data, //amount of work               0
             const int use_index,//flag for index calculation   1
             const int use_hess, //flag for hessian             2
    __global const int* data,    //data pointer                 3      0
    __global const int* index,   //index pointer (might be 0)   4      1
    __global const float* grad,  //gradient input pointer       5      2
    __global const float* hess,  //hessian input pointer        6      3
    __global       int* count,   //counter output               7      4    0
    __global       float* hgrad, //gradient output              8      5    1
    __global       float* hhess, //hessian output               9      6    2
    //local chaches
    __local        int*  Lhist,  //NUM_BINS*loc_size           10
    __local        float* Lgrad, //NUM_BINS*loc_size           11
    __local        float* Lhess, //NUM_BINS*loc_size           12
    __local        int*  Lidx    //numdata                     13
    )
{
    //work_item ids
    __private int my_id = get_global_id(0);
    __private int loc_id = get_local_id(0);
    __private int loc_size = get_local_size(0);
    //__private int loc_size = 512;
    __private int grp_id = get_group_id(0);

    //amount of work for one work_item
    __private int shift = num_data / loc_size;

    //local storage
    // __local int Lhist[NUM_BINS*loc_size];
    // __local float Lgrad[NUM_BINS*loc_size];
    // __local float Lhess[NUM_BINS*loc_size];
    // __local int Lidx[num_data];

    //private vars for loops, data pointing
    __private int j = 0;
    __private int bin = 0;
    __private int i = 0;

    /* work-item/memory map
           0     k    2k    3k    4k    5k    6k    7k    8k    9k   10k   11k   12k  << k is shift
    array  |-----------------------------------------------------------------------|
    loc_id |--0--|--1--|--2--|--3--|--4--|--5--|--6--|--7--|--8--|--9--|-10--|-11--|  << assume loc_size = 12
    grp_id \----------------------------------0------------------------------------/

    */
    //if (my_id==0){printf("count [0] = %d\n", count[0]);}
    //saving data
    if (my_id < NUM_BINS){
        __attribute__((xcl_pipeline_loop))
        for (j = my_id; j < NUM_BINS; j += loc_size){
            Lhist[j] = count[j];
            Lgrad[j] = hgrad[j];
            Lhess[j] = hhess[j];
            //printf("{Thr %d [%d] count=%d hgrad=%f hhess=%f}",my_id,j,count[j],hgrad[j],hhess[j]);
            //Lhist[j] = 0;
            //Lgrad[j] = 0;
            //Lhess[j] = 0;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE); //sync all

    /*
    my_id 0 1 2 3 4 5 0 1 2 3  4  5 ...
    shift <---------> <-----------> ...
    Lidx  0 1 2 3 4 5 6 7 8 9 10 11 ...
    */

    //getting indexes
    if (use_index == 1){
        //load from index
        __attribute__((xcl_pipeline_loop))
        for (j = 0; j < shift; j ++){
            Lidx[j*loc_size + my_id] = index[j*loc_size + my_id];
        }
    } else {
        __attribute__((xcl_pipeline_loop))
        for (j = 0; j < shift; j ++){
            Lidx[j*loc_size + my_id] = j*loc_size + my_id;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE); //sync all
    //calc histogram locally

    /*
    histbin 0 0 0 0 0 0 ...        0 1 1 ...
    my_id   0 1 2 3 4 5 ... loc_size 0 1 ...
    */

    __attribute__((xcl_pipeline_loop))
    for (j=0; j < shift; j++){
        i = Lidx[my_id + j*loc_size];
        bin = data[i];
        i = my_id*NUM_BINS+bin;
        Lhist[i]++;
        // if (bin == 49){
        //     printf("{HIST THR %d bin %d STRT j=%d, i=%d, Lgrad[i] = %f, add grad[%d] = %f}\n", my_id, bin, j, i, Lgrad[i], my_id + j*loc_size, grad[my_id + j*loc_size]);
        // }
        Lgrad[i] += grad[my_id + j*loc_size];
        // if (bin == 49){
        //     printf("{HIST THR %d bin %d END Lgrad[%d] = %f}\n", my_id, bin, i, Lgrad[i]);
        // }
    }
    if (use_hess == 1){
        __attribute__((xcl_pipeline_loop))
        for (j=0; j < shift; j++){
            i = Lidx[my_id + j*loc_size];
            bin = data[i];
            Lhess[my_id*NUM_BINS+bin] += hess[my_id + j*loc_size];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //reduce to get final histogram

    /*
    Have 16 histograms with 64 bins
    */

    if (my_id < NUM_BINS){
        __attribute__((xcl_pipeline_loop))
        for (i=my_id; i < NUM_BINS; i += loc_size){
            for (j=1; j < loc_size; j++){
                Lhist[i] += Lhist[j*NUM_BINS+i];
                if (i == 49){
                    printf("{REDC THR %d bin %d STRT j=%d, i=bin, Lgrad[i] = %f, add Lgrad[%d] = %f}\n", my_id, i, j, Lgrad[i], my_id + j*loc_size, j*NUM_BINS+i, Lgrad[j*NUM_BINS+i]);
                }
                Lgrad[i] += Lgrad[j*NUM_BINS+i];
                if (i == 49){
                    printf("{REDC THR %d bin %d END Lgrad[bin] = %f}\n", my_id, i, Lgrad[i]);
                }
            }
        }
        if (use_hess == 1){
            __attribute__((xcl_pipeline_loop))
            for (i=my_id; i < NUM_BINS; i += loc_size){
                for (j=0; j < loc_size; j++){
                    Lhess[i] += Lhess[j*NUM_BINS+i];
                }
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //writing to output
    if (my_id < NUM_BINS){
        __attribute__((xcl_pipeline_loop))
        for (j = my_id; j < NUM_BINS; j += loc_size){
            //printf("Thread %d saving %d to count[%d] rewriting %d \n", my_id, Lhist[i], j, count[j]);
            count[j] = Lhist[j];
            printf("{OUT THR %d hgrad[%d]==%f}\n", my_id, j, hgrad[j]);
            hgrad[j] = Lgrad[j];
            if (use_hess == 1){
                hhess[j] = Lhess[j];
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}
