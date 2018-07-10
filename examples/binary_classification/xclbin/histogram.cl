// This function represents an OpenCL kernel. The kernel will be call from
// host application using the xcl_run_kernels call. The pointers in kernel
// parameters with the global keyword represents cl_mem objects on the FPGA
// DDR memory.

#define HESSSIANS   0x1
#define GRADIENTS   0x2
#define DATAINDICES 0x4

//kernel __attribute__((reqd_work_group_size(DATASIZE, 1, 1)))
//test function
void kernel hist_add(global       int* hist,  //output
                     global const int* data,  //input - bins
                            const int  nblock) //amount of items per kernel
                     //const int  Nbins     //setup - number of bins in histogram
{
    int myid = get_global_id(0);
    for (int i = 0; i < nblock; i++)
    {
      int bin = data[nblock * (myid - 1) + i];
      //printf("Kernel %d processing item %d (out of %d) at bin %d from addr %d\n", myid, i+1, nblock, bin, nblock * (myid - 1) + i);
      hist[bin] = hist[bin] + 1;
    }
}

///LIGHT GBM

void kernel hist_lightgbm(       const int    num_data,          //amount of data per one instance of a kernel
                          global const int*   data,              //input bin indices
                          global const int*   data_indices,      //input data indices
                          global const float* gradients,         //input gradients
                          global const float* hessians,          //input hessians
                          global       float* out_gradients,     //output sum gradients
                          global       float* out_hessians,      //output sum hessians
                          global       int*   out_counter,       //output histogram (should not be NULL)
                                 const int    mode,              //sets what to use
                                 const int    debug)             //print debug messages
{
    //mode
    // 0x1 - use hassians
    // 0x2 - use gradients
    // 0x4 - use data_indices
    int myid = get_global_id(0);
    if (debug == 1){
        for (int i = 0; i < 10; i++){
            printf("Kernel %d data[%d] = %d\n", myid, i, data[i]);
        }
        if (mode & DATAINDICES){
            for (int i = 0; i < 10; i++){
                printf("Kernel %d data_index[%d] = %d\n", myid, i, data_indices[i]);
            }
        }
        printf("Kernel %d arguments: num_data %d, mode %d\n", myid, num_data, mode);
    }
    for (int i = 0; i < num_data; i++)
    {
        int idx = num_data * myid + i;
        if (mode & DATAINDICES) {
            idx = data_indices[idx];
        }

        int bin = data[idx];

        //DEBUG for targets "sw_emu" and "hw_emu"
        if (debug == 1){
            printf("Kernel %d mode %d processing item %d (out of %d) at bin %d from index %d.\n", myid, mode, i, num_data-1, bin, idx);
            printf("Kernel %d bin %d has value %d, gradient %f hessian %f\n", myid, bin, out_counter[bin], out_hessians[bin], out_gradients[bin]);
        }

        out_counter[bin] = out_counter[bin] + 1;

        if (mode & HESSSIANS){
            out_hessians[bin] = out_hessians[bin] + hessians[idx];
        }

        if (mode & GRADIENTS){
            out_gradients[bin] = out_gradients[bin] + gradients[idx];
        }
    }
}
