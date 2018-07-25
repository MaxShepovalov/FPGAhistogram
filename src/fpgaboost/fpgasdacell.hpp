//fpgasdacell.hpp

#ifndef FPGASDACELL
#define FPGASDACELL

#include <vector>
#include <iostream>
#include <LightGBM/meta.h>
#include <mutex>

#include <typeinfo>

//fpga connection libraries
#include "xcl2.hpp"
#include <vector>
#include <cstdlib>

using std::vector;

//mode switches
#define HESSIANS   0x1
#define GRADIENTS   0x2
#define DATAINDICES 0x4

#define ONETHREAD //allow only one thread per fpgacall

//#define USE_OLD_FPGACALL //use "always-load-parameters" fpgacall

void check(cl_int err, int linenum) {
  if (err) {
    printf("ERROR at line %d: Operation Failed: %d\n", linenum, err);
    exit(EXIT_FAILURE);
  }
}

#ifdef FPGADEBUG
#pragma message "\n\n      FPGADEBUG is active \n\n"

template <typename T>
void printArray(std::vector<T, std::allocator<T>> data, int size, const char* name){
    if (size >= 1) {
        //printf("%s = [%d", name, data[0]);
        std::cout << name << " = [" << data[0];
        for (int i=1; i < size; i++){
            //printf(", %d", data[i]);
            std::cout << ", " << data[i];
        }
        //printf("]\n");
        std::cout << "]\n";
    } else {
        //printf("%s []\n", name);
        std::cout << name << " []\n";
    }
    
}
template <typename T>
void printArray(std::vector<T, aligned_allocator<T> > data, int size, const char* name){
    if (size >= 1) {
        //printf("%s = [%d", name, data[0]);
        std::cout << name << " = [" << data[0];
        for (int i=1; i < size; i++){
            //printf(", %d", data[i]);
            std::cout << ", " << data[i];
        }
        //printf("]\n");
        std::cout << "]\n";
    } else {
        //printf("%s []\n", name);
        std::cout << name << " []\n";
    }
    
}

template <typename T>
void printArray(T* data, int size, const char* name){
    if (size >= 1) {
        //printf("%s = [%d", name, data[0]);
        std::cout << name << " = [" << data[0];
        for (int i=1; i < size; i++){
            //printf(", %d", data[i]);
            std::cout << ", " << data[i];
        }
        //printf("]\n");
        std::cout << "]\n";
    } else {
        //printf("%s []\n", name);
        std::cout << name << " []\n";
    }
    
}

template <typename T>
void printArray(const T* data, int size, const char* name){
    if (size >= 1) {
        //printf("%s = [%d", name, data[0]);
        std::cout << name << " = [" << data[0];
        for (int i=1; i < size; i++){
            //printf(", %d", data[i]);
            std::cout << ", " << data[i];
        }
        //printf("]\n");
        std::cout << "]\n";
    } else {
        //printf("%s []\n", name);
        std::cout << name << " []\n";
    }
    
}
#endif

//cl::Context context;
//cl::CommandQueue q;
//cl::Program program;
//buffers for moving data
// cl::Buffer buffer_bins = cl::Buffer();
// cl::Buffer buffer_inds = cl::Buffer();
// cl::Buffer buffer_grad = cl::Buffer();
// cl::Buffer buffer_hess = cl::Buffer();
// cl::Buffer buffer_hist = cl::Buffer();
// cl::Buffer buffer_sgrad = cl::Buffer();
// cl::Buffer buffer_shess = cl::Buffer();

//std::vector<cl::Memory> binsBufVec;
//std::vector<cl::Memory> indsBufVec;
//std::vector<cl::Memory> gradBufVec;
//std::vector<cl::Memory> hessBufVec;
//std::vector<cl::Memory> histBufVec;
//std::vector<cl::Memory> sgradBufVec;
//std::vector<cl::Memory> shessBufVec;

//previous arguments for fpgacall. Used to decide if CPU->FPGA transfer needed
//int          last_data_size = 0;
// void* last_data_pointer = 0;

// float* last_hessian_pointer = 0;
// float* last_gradient_pointer = 0;
// //int          last_index_size = 0;
// int*   last_index_pointer = 0;
// //int          last_histogram_size = 0;
// float*       last_histogram_hessian = 0;
// float*       last_histogram_gradient = 0;
// int*         last_histogram_couter = 0;
// //int          last_mode = 0;

//thread control
std::mutex mtx;

// bool device_setup = false;
#ifdef FPGADEBUG
bool dbg = true;
#else
bool dbg = false;
#endif

 cl::Program init_xilinx(cl::CommandQueue & q, cl::Context & context){}
//     //mtx.lock(); //need to init FPGA only once

//     cl::Program program;

//     if (!device_setup){

//     //XILINX FPGA SETUP
//     // The get_xil_devices will return vector of Xilinx Devices 
//     std::vector<cl::Device> devices = xcl::get_xil_devices();
//     cl::Device device = devices[0];

//     //Creating Context and Command Queue for selected Device 
//     //cl::Context context(device);
//     context = cl::Context(device);
//     //cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);
//     q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
//     std::string device_name = device.getInfo<CL_DEVICE_NAME>();

//     // the OpenCL binary file created using the 
//     // xocc compiler load into OpenCL Binary and return as Binaries
//     // OpenCL and it can contain many functions which can be executed on the
//     // device.
//     std::string binaryFile = xcl::find_binary_file(device_name,"histogram");
//     cl::Program::Binaries binary_file = xcl::import_binary_file(binaryFile);
//     devices.resize(1);
//     program = cl::Program(context, devices, binary_file);

// #ifdef FPGADEBUG
//     std::cout << "Found Device=" << device_name.c_str() << std::endl;
//     size_t max_workgroup_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
//     printf("Device's max_workgroup_size is %ld\n", max_workgroup_size);
// #endif

//     device_setup = true;
//     }

//     //mtx.unlock();
//     return program;
// }

// template <typename T>
// int load_vector_to_buffer(cl::CommandQueue q, cl::Context context, int size, vector<T,aligned_allocator<T>> host_vector, std::vector<cl::Memory> & BufVec){
// #ifdef FPGADEBUG
//     //printf("Loading of %d values to device\n", size);
//     std::cout << "Loading of " << size << " values of type " << typeid(T).name() << " to device." << std::endl;
// #endif
//     size_t size_in_bytes = size * sizeof(T);
//     BufVec.clear();
//     cl::Buffer buf = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, size_in_bytes, host_vector.data());
//     BufVec.push_back(buf); // CL_MEM_ALLOC_HOST_PTR
//     cl_int err = q.enqueueMigrateMemObjects(BufVec,0);
//     q.finish(); //block write
//     //cl_int err = q.enqueueWriteBuffer(buf, CL_TRUE, 0, size_in_bytes, static_cast<void*>(host_vector.data()));
//     if (err != 0 || dbg){
//         printf("MIGRATEMEMOBJECTS (host -> device) code: %d, vector @ %ld with size %d\n", err, &BufVec, BufVec.size());
//         //printf("MIGRATEMEMOBJECTS (host -> device) code: %d, vector @ %ld\n", err, &BufVec);
//     }
//     return err;
// }

// template <typename T>
// int load_vector_to_buffer(cl::CommandQueue q, cl::Context context, int size, std::vector<T, std::allocator<T>> host_vector, std::vector<cl::Memory> & BufVec){
// #ifdef FPGADEBUG
//     std::cout << "Converting vector of size " << size << " to be alligned for next transfer" << std::endl;
// #endif
//     vector<int,aligned_allocator<int>> alligned_host_vector;
//     for (int i = 0; i < size; i++){
//         alligned_host_vector.insert(alligned_host_vector.begin() + i, (int)host_vector[i]);
//     }
//     int err = load_vector_to_buffer<int>(q, context, size, alligned_host_vector, BufVec);
//     return err;
// }

// template <typename T>
// int load_array_to_buffer(cl::CommandQueue q, cl::Context context, int size, const T* host_array, std::vector<cl::Memory> & BufVec){
//     //vector<T,aligned_allocator<T>> host_data(host_array, host_array + size);
//     vector<T,aligned_allocator<T>> host_data;
//     for (int i=0; i < size; i++){
//         host_data.insert(host_data.begin() + i, host_array[i]);
//     }
//     int err = load_vector_to_buffer<T>(q, context, size, host_data, BufVec);
//     return err;
// }

//#ifndef USE_OLD_FPGACALL

// template <typename VAL_T>
// int fpgacall_onethread(
//              int    data_size,
//              const std::vector<VAL_T, std::allocator<VAL_T>> data_pointer,
//              const float* hessian_pointer,
//              const float* gradient_pointer,
//              int    index_size,
//              const int*   index_pointer,
//              int    histogram_size,
//              float* histogram_hessian,
//              float* histogram_gradient,
//              int*   histogram_couter,
//              int    mode){

// #ifdef FPGADEBUG
//     printf("DEBUG VERSION OF FPGASDACELL.HPP IS RUNNING\n");
// #endif

//     bool precheck = true;
//     int work_size = mode & DATAINDICES ? index_size : data_size;

//     if ((mode & GRADIENTS) && !gradient_pointer){
//         printf("fgpasdacell: Mode require gradients, but gradient_pointer is NULL\n");
//         precheck = false;
//     }
//     if ((mode & HESSIANS) && !hessian_pointer){
//         printf("fgpasdacell: Mode require hessians, but hessian_pointer is NULL\n");
//         precheck = false;
//     }
//     if ((mode & DATAINDICES) && !gradient_pointer){
//         printf("fgpasdacell: Mode require indices, but index_pointer is NULL\n");
//         precheck = false;
//     }
//     //if (!device_setup){
//     //    printf("fpgasdacell: init device before fpgasdacell()\n");
//     //    precheck = false;
//     //}
//     if (!precheck){
//         printf("fgpasdacell: Wrong input parameters\n");
//         return -2;
//     }
//     if (work_size==0){
//         //printf("WORK_SIZE IS ZERO!!\n");
//         return 0;
//     }

//     //output
//     vector<int,aligned_allocator<int>> histogram(histogram_couter, histogram_couter + histogram_size);
//     vector<float,aligned_allocator<float>> sum_gradients(histogram_gradient, histogram_gradient + histogram_size);
//     vector<float,aligned_allocator<float>> sum_hessians(histogram_hessian, histogram_hessian + histogram_size);

//     //patches for unused arrays
//     vector<int,aligned_allocator<int>> intpatch(1, 0);
//     vector<float,aligned_allocator<float>> floatpatch(1, 0.0);

//     device_setup = false;
//     cl::CommandQueue q;
//     cl::Context context;
//     cl::Program program = init_xilinx(q, context);

//     bool datafailed = false;
//     //load input
//     if ((void*)&data_pointer != last_data_pointer){
// #ifdef FPGADEBUG
//         printf("New data pointer %ld (was %ld)\n", (void*)&data_pointer, last_data_pointer);
// #endif
//         load_vector_to_buffer<VAL_T>(q, context, data_size, data_pointer, binsBufVec);
//         last_data_pointer = (void*)&data_pointer;
//     }
//     if (binsBufVec.size() == 0){
//         printf("FPGA's buffer for data_ is empty");
//         return 0;
//     }

//     if (mode & DATAINDICES) {
//         if (index_pointer != last_index_pointer and index_pointer != NULL){
// #ifdef FPGADEBUG
//             printf("New index pointer %ld (was %ld)\n", index_pointer, last_index_pointer);
// #endif
//             int err = load_array_to_buffer<int>(q, context, index_size, index_pointer, indsBufVec);
//             last_index_pointer = const_cast<int*>(index_pointer);
//             if (err != 0){
//                 datafailed = true;
//             }
//         }
//     }
//     if (indsBufVec.size() == 0){
//         int err = load_vector_to_buffer<int>(q, context, 1, intpatch, indsBufVec);
//         if (err != 0){
//             datafailed = true;
//         }
//     }

//     if (mode & GRADIENTS) {
//         if (gradient_pointer != last_gradient_pointer and gradient_pointer != NULL){
// #ifdef FPGADEBUG
//             printf("New gradients pointer %ld (was %ld)\n", gradient_pointer, last_gradient_pointer);
// #endif
//             int err = load_array_to_buffer<float>(q, context, data_size, gradient_pointer, gradBufVec);
//             last_gradient_pointer = const_cast<float*>(gradient_pointer);
//             if (err != 0){
//                 datafailed = true;
//             }
//         }
//     }
//     if (gradBufVec.size() == 0){
//         int err = load_vector_to_buffer<float>(q, context, 1, floatpatch, gradBufVec);
//         if (err != 0){
//             datafailed = true;
//         }
//     }

//     if (mode & HESSIANS) {
//         if (hessian_pointer != last_hessian_pointer and hessian_pointer != NULL) {
// #ifdef FPGADEBUG
//             printf("New hessians pointer %ld (was %ld)\n", hessian_pointer, last_hessian_pointer);
// #endif
//             int err = load_array_to_buffer<float>(q, context, data_size, hessian_pointer, hessBufVec);
//             last_hessian_pointer = const_cast<float*>(hessian_pointer);
//             if (err != 0){
//                 datafailed = true;
//             }
//         }
//     }
//     if (hessBufVec.size() == 0){
//         int err = load_vector_to_buffer<float>(q, context, 1, floatpatch, hessBufVec);
//         if (err != 0){
//             datafailed = true;
//         }
//     }

//     //load output
//     if (last_histogram_couter != histogram_couter){
// #ifdef FPGADEBUG
//         printf("New histogram counter pointer %ld (was %ld)\n", histogram_couter, last_histogram_couter);
// #endif
//         int err = load_vector_to_buffer<int>(q, context, histogram_size, histogram, histBufVec);
//         last_histogram_couter = histogram_couter;
//         if (err != 0){
//             datafailed = true;
//         }
//     }
//     if (last_histogram_gradient != histogram_gradient){
// #ifdef FPGADEBUG
//         printf("New histogram gradients pointer %ld (was %ld)\n", histogram_gradient, last_histogram_gradient);
// #endif
//         int err = load_vector_to_buffer<float>(q, context, histogram_size, sum_gradients, sgradBufVec);
//         last_histogram_gradient = histogram_gradient;
//         if (err != 0){
//             datafailed = true;
//         }
//     }
//     if (last_histogram_hessian != histogram_hessian){
// #ifdef FPGADEBUG
//         printf("New histogram hessians pointer %ld (was %ld)\n", histogram_hessian, last_histogram_hessian);
// #endif
//         int err = load_vector_to_buffer<float>(q, context, histogram_size, sum_hessians, shessBufVec);
//         last_histogram_hessian = histogram_hessian;
//         if (err != 0){
//             datafailed = true;
//         }
//     }

//     if (datafailed){
//         printf("Data transfer failed\n");
//         exit(-1);
//     }

// #ifdef FPGADEBUG
//         printf("=====\nINPUTS:\n");
//         //printArray<VAL_T>(bins, data_size, "data");
//         printArray<VAL_T>(data_pointer, data_size, "data");
//         printArray<int>(index_pointer, index_size, "indices");
//         printArray<float>(gradient_pointer, data_size, "gradients");
//         printArray<float>(hessian_pointer, data_size, "hessians");
//         printf("=====/inputs\n");
// #endif

//     // extract a kernel
//     cl::Kernel krnl_hist_add(program, "hist_lightgbm");

//     //set the kernel Arguments
//     int narg=0;
//     int items_per_call = 1;
//     int work_size_ocl = work_size / items_per_call;
//     for (int i=256; i >= 1; i = i / 2){
//         if (work_size % i == 0){
//             items_per_call = i;
//             work_size_ocl = work_size / i;
//             break;
//         }
//     }
//     krnl_hist_add.setArg(narg++, items_per_call);//amount of data per one instance of a kernel
//     krnl_hist_add.setArg(narg++, binsBufVec[0]);//input bin indices
//     krnl_hist_add.setArg(narg++, indsBufVec[0]);//input data indices
//     krnl_hist_add.setArg(narg++, gradBufVec[0]);//input gradients
//     krnl_hist_add.setArg(narg++, hessBufVec[0]);//input hessians
//     krnl_hist_add.setArg(narg++, sgradBufVec[0]);//output sum gradients
//     krnl_hist_add.setArg(narg++, shessBufVec[0]);//output sum hessians
//     krnl_hist_add.setArg(narg++, histBufVec[0]);//output histogram (should not be NULL)
//     krnl_hist_add.setArg(narg++, mode);//sets what to use
// #ifdef FPGADEBUG
//     krnl_hist_add.setArg(narg++, 1);//do debug print
//     printf("Kernel arguments should be set. Glob Work_size %d, work load per kernel %d, Total items %d\n", work_size_ocl, items_per_call, work_size);
// #else
//     krnl_hist_add.setArg(narg++, 0);//skip debug print
// #endif

//     //Launch the Kernel
//     q.enqueueNDRangeKernel(
//         krnl_hist_add, //kernel
//         cl::NullRange,    //work_dim (offset)
//         cl::NDRange(work_size_ocl),   //work_size (global)
//         cl::NullRange     //work_group_size (local)
//         );

//     // The result of the previous kernel execution will need to be retrieved in
//     // order to view the results. This call will write the data from the
//     // buffer_result cl_mem object to the source_results vector
//     cl_int err = q.enqueueMigrateMemObjects(histBufVec, CL_MIGRATE_MEM_OBJECT_HOST);
//     if (err != 0 || dbg){
//         printf("MIGRATEMEMOBJECTS histBufVec (host <- device) code: %d\n", err);
//     }
    
//     if (mode & HESSIANS){
//         cl_int err = q.enqueueMigrateMemObjects(shessBufVec, CL_MIGRATE_MEM_OBJECT_HOST);
//         if (err != 0 || dbg){
//             printf("MIGRATEMEMOBJECTS shessBufVec (host <- device) code: %d\n", err);
//         }
//     }
//     if (mode & GRADIENTS){
//         cl_int err = q.enqueueMigrateMemObjects(sgradBufVec, CL_MIGRATE_MEM_OBJECT_HOST);
//         if (err != 0 || dbg){
//             printf("MIGRATEMEMOBJECTS sgradBufVec (host <- device) code: %d\n", err);
//         }
//     }

//     q.finish();

// #ifdef FPGADEBUG
//         printf("====RESULTS\n");
//         printf("mode = %d\n", mode);
//         printArray<int>(histogram, histogram_size, "hist");
//         printArray<float>(sum_gradients, histogram_size, "sum_grad");
//         printArray<float>(sum_hessians, histogram_size, "sum_hess");
// #endif

//     //move out data
//     for (int i=0; i < histogram_size; i++){
//         histogram_couter[i] = histogram[i];
//         if (mode & GRADIENTS){
//             histogram_gradient[i] = sum_gradients[i];
//         }
//         if (mode & HESSIANS){
//             histogram_hessian[i] = sum_hessians[i];
//         }
//     }

//     return err;
// }

// template <typename VAL_T>
// int fpgacall(
//              int    data_size,
//              const std::vector<VAL_T, std::allocator<VAL_T>> data_pointer,
//              const float* hessian_pointer,
//              const float* gradient_pointer,
//              int    index_size,
//              const int*   index_pointer,
//              int    histogram_size,
//              float* histogram_hessian,
//              float* histogram_gradient,
//              int*   histogram_couter,
//              int    mode){
    
//     mtx.lock();

//     int err = fpgacall_onethread(
//         data_size, data_pointer, hessian_pointer, gradient_pointer,
//         index_size, index_pointer, histogram_size, histogram_hessian,
//         histogram_gradient, histogram_couter, mode);
    
//     mtx.unlock();

//     return err;
// }


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//#else

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename VAL_T>
int fpgacall(
             int    data_size,
             const std::vector<VAL_T, std::allocator<VAL_T>> data_pointer,
             const float* hessian_pointer,
             const float* gradient_pointer,
             int    index_size,
             const int*   index_pointer,
             int    histogram_size,
             float* histogram_hessian,
             float* histogram_gradient,
             int*   histogram_couter,
             int    mode){

#ifdef ONETHREAD
    mtx.lock();
#endif

    bool precheck = true;
    int work_size = mode & DATAINDICES ? index_size : data_size;

    if ((mode & GRADIENTS) && !gradient_pointer){
        printf("fgpasdacell: Mode require gradients, but gradient_pointer is NULL\n");
        precheck = false;
    }
    if ((mode & HESSIANS) && !hessian_pointer){
        printf("fgpasdacell: Mode require hessians, but hessian_pointer is NULL\n");
        precheck = false;
    }
    if ((mode & DATAINDICES) && !gradient_pointer){
        printf("fgpasdacell: Mode require indices, but index_pointer is NULL\n");
        precheck = false;
    }
    // if (!device_setup){
    //     printf("fpgasdacell: init device before fpgasdacell()\n");
    //     precheck = false;
    // }
    if (!precheck){
        printf("fgpasdacell: Wrong input parameters\n");
        return -2;
    }
    if (work_size==0){
        //printf("WORK_SIZE IS ZERO!!\n");
        return 0;
    }

/////////////////////////////////////////////////////////////////////////////////////////////
    cl::Program program;

    //if (!device_setup){

    //XILINX FPGA SETUP
    // The get_xil_devices will return vector of Xilinx Devices 
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    //Creating Context and Command Queue for selected Device 
    cl::Context context(device);
    //context = cl::Context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);
    //q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();

    // the OpenCL binary file created using the 
    // xocc compiler load into OpenCL Binary and return as Binaries
    // OpenCL and it can contain many functions which can be executed on the
    // device.
    std::string binaryFile = xcl::find_binary_file(device_name,"histogram");
    cl::Program::Binaries binary_file = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    program = cl::Program(context, devices, binary_file);
/////////////////////////////////////////////////////////////////////////////////////////////

    //compute the size of array in bytes

    // Creates a vector for data
    //input
    //data_pointer is a vector already. //vector<int,aligned_allocator<int>> bins(data_size, 0);
    vector<int,aligned_allocator<int>> indices(index_size, 0);
    vector<float,aligned_allocator<float>> gradients(data_size, 0.0);
    vector<float,aligned_allocator<float>> hessians(data_size, 0.0);
    //output
    vector<int,aligned_allocator<int>> histogram(histogram_size, 0);
    vector<float,aligned_allocator<float>> sum_gradients(histogram_size, 0.0);
    vector<float,aligned_allocator<float>> sum_hessians(histogram_size, 0.0);

    //patches for unused arrays
    vector<int,aligned_allocator<int>> intpatch(1, 0);
    vector<float,aligned_allocator<float>> floatpatch(1, 0.0);

    //device should be set up already
    
    //fill data
    //srand ( time(NULL) );
    for (int i=0; i < data_size; i++){
        //int value = data_pointer[i];
        //bins.insert(bins.begin() + i, value);

        if (mode & HESSIANS) {
            hessians.insert(hessians.begin() + i, hessian_pointer[i]);
        }

        if (mode & GRADIENTS) {
            gradients.insert(gradients.begin() + i, gradient_pointer[i]);
        }
    }
    if (mode & DATAINDICES) {
        for (int i=0; i< index_size; i++){
            int value = index_pointer[i];
            indices.insert(indices.begin() + i, value);
        }
    }
    for (int i=0; i < histogram_size; i++){
        histogram.insert(histogram.begin() + i, histogram_couter[i]);

        if (mode & HESSIANS) {
            hessians.insert(hessians.begin() + i, histogram_hessian[i]);
        }

        if (mode & GRADIENTS) {
            gradients.insert(gradients.begin() + i, gradient_pointer[i]);
        }
    }

#ifdef FPGADEBUG
        printf("=OLD_FPGGACALL_VERSION====\nINPUTS:\n");
        //printArray<VAL_T>(bins, data_size, "data");
        printArray<VAL_T>(data_pointer, data_size, "data");
        printArray<int>(indices, index_size, "indices");
        printArray<float>(gradients, data_size, "gradients");
        printArray<float>(hessians, data_size, "hessians");
        VAL_T k;
        printf("Size of VAL_T: %d and type of %s\n=====\n", sizeof(VAL_T), typeid(k).name());
        unsigned char uc;
        printf("Size of unsigned char: %d type %s\n", sizeof(unsigned char), typeid(uc).name());
        unsigned short s;
        printf("Size of unsigned short: %d and type of %s\n", sizeof(unsigned short), typeid(s).name());
        unsigned int r;
        printf("Size of unsigned int: %d and type of %s\n", sizeof(unsigned int), typeid(r).name());
        printf("=====/inputs\n");
#endif

    // These commands will allocate memory on the FPGA
    //input
    cl::Buffer buffer_bins;
    cl::Buffer buffer_inds;
    cl::Buffer buffer_grad;
    cl::Buffer buffer_hess;
    // cl::Buffer buffer_bins(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, data_size_in_bytes, bins.data());

    size_t data_size_in_bytes = data_pointer.size() * sizeof(VAL_T);
    size_t grad_size_in_bytes = gradients.size() * sizeof(float);
    size_t hess_size_in_bytes = hessians.size() * sizeof(float);
    size_t index_size_in_bytes = index_size * sizeof(int);
    size_t hist_size_in_bytes = histogram_size * sizeof(int);
    size_t sums_size_in_bytes = histogram_size * sizeof(float);
    
    buffer_bins = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, data_size_in_bytes, const_cast<VAL_T*>(data_pointer.data()));
    if (mode & DATAINDICES) {
        buffer_inds = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, index_size_in_bytes, indices.data());
    } else {
        buffer_inds = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 1, intpatch.data());
    }
    if (mode & GRADIENTS) {
        buffer_grad = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, grad_size_in_bytes, gradients.data());
    } else {
        buffer_grad = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 1, floatpatch.data());
    }
    if (mode & HESSIANS) {
        buffer_hess = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hess_size_in_bytes, hessians.data());
    } else {
        buffer_hess = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 1, floatpatch.data());
    }
    //output
    cl::Buffer buffer_hist(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, hist_size_in_bytes, histogram.data());
    cl::Buffer buffer_sgrad(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sums_size_in_bytes, sum_gradients.data());
    cl::Buffer buffer_shess(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sums_size_in_bytes, sum_hessians.data());
    
    //Separate Read/write Buffer vector is needed to migrate data between host/device
    std::vector<cl::Memory> inBufVec, outBufVec;
    inBufVec.push_back(buffer_bins);
    inBufVec.push_back(buffer_inds);
    inBufVec.push_back(buffer_grad);
    inBufVec.push_back(buffer_hess);

    outBufVec.push_back(buffer_hist);
    outBufVec.push_back(buffer_sgrad);
    outBufVec.push_back(buffer_shess);

    // load data vector from the host to the FPGA
    cl_int err = q.enqueueMigrateMemObjects(inBufVec,0/* 0 means from host*/);
    check(err, 727);

#ifdef FPGADEBUG
        printf("MIGRATEMEMOBJECTS inBufVec (host -> device) code: %d\n", err);
#endif

    err = q.enqueueMigrateMemObjects(outBufVec,0);
    check(err, 734);

#ifdef FPGADEBUG
        printf("MIGRATEMEMOBJECTS outBufVec (host -> device) code: %d\n", err);
#endif

    // extract a kernel
    cl::Kernel krnl_hist_add(program, "hist_lightgbm");

    //set the kernel Arguments
    int narg=0;
    int items_per_call = 1;
    krnl_hist_add.setArg(narg++, items_per_call);//amount of data per one instance of a kernel
    krnl_hist_add.setArg(narg++, buffer_bins);//input bin indices
    krnl_hist_add.setArg(narg++, buffer_inds);//input data indices
    krnl_hist_add.setArg(narg++, buffer_grad);//input gradients
    krnl_hist_add.setArg(narg++, buffer_hess);//input hessians
    krnl_hist_add.setArg(narg++, buffer_sgrad);//output sum gradients
    krnl_hist_add.setArg(narg++, buffer_shess);//output sum hessians
    krnl_hist_add.setArg(narg++, buffer_hist);//output histogram (should not be NULL)
    krnl_hist_add.setArg(narg++, mode);//sets what to use
#ifdef FPGADEBUG
    krnl_hist_add.setArg(narg++, 1);//do debug print
#else
    krnl_hist_add.setArg(narg++, 0);//skip debug print
#endif

#ifdef FPGADEBUG
        printf("Kernel arguments should be set\n");
#endif

    //Launch the Kernel
    q.enqueueNDRangeKernel(
        krnl_hist_add, //kernel
        {1},    //work_dim (offset)
        {work_size},   //work_size (global)
        {1}     //work_size (local)
        );

    // The result of the previous kernel execution will need to be retrieved in
    // order to view the results. This call will write the data from the
    // buffer_result cl_mem object to the source_results vector
    err = q.enqueueMigrateMemObjects(outBufVec,CL_MIGRATE_MEM_OBJECT_HOST);
    check(err, 777);

#ifdef FPGADEBUG
        printf("MIGRATEMEMOBJECTS outBufVec (host <- device) code: %d\n", err);
#endif

    q.finish();

#ifdef FPGADEBUG
        printf("====RESULTS\n");
        printf("mode = %d\n", mode);
        printArray<int>(histogram, histogram_size, "hist");
        printArray<float>(sum_gradients, histogram_size, "sum_grad");
        printArray<float>(sum_hessians, histogram_size, "sum_hess");
#endif

    //move out data
    for (int i=0; i < histogram_size; i++){
        histogram_couter[i] = histogram[i];
        if (mode & GRADIENTS){
            histogram_gradient[i] = sum_gradients[i];
        }
        if (mode & HESSIANS){
            histogram_hessian[i] = sum_hessians[i];
        }
    }

#ifdef ONETHREAD
    mtx.unlock();
#endif
    return err;
}


//#endif //USE_OLD_FPGACALL

#endif //FPGASDACELL
