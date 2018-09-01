//fpgasdacell.hpp

#ifndef FPGASDACELL
#define FPGASDACELL
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#include <vector>
#include <iostream>
#include <LightGBM/meta.h>
#include <mutex>

#include <typeinfo>

//fpga connection libraries
//#include "xcl2.hpp"
#include "xcl.h"
#include <CL/cl.h>
#include <vector>
#include <cstdlib>

using std::vector;

//mode switches
#define HESSIANS   0x1
#define GRADIENTS   0x2
#define DATAINDICES 0x4

#define ONETHREAD //allow only one thread per fpgacall

//#define USE_OLD_FPGACALL //use "always-load-parameters" fpgacall

template <class T> const T& min (const T& a, const T& b) {
  return !(b<a)?a:b;     // or: return !comp(b,a)?a:b; for version (2)
}

// void check(cl_int err, int linenum) {
//   if (err) {
//     printf("ERROR at line %d: Operation Failed: %d\n", linenum, err);
//     exit(EXIT_FAILURE);
//   }
// #ifdef FPGADEBUG
//   printf("Line %d status: %d\n", linenum, err);
// #endif
// }

uint64_t get_duration_ns (const cl::Event &event) {
    uint64_t nstimestart, nstimeend;
    event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START,&nstimestart);
    event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END,&nstimeend);
    return(nstimeend-nstimestart);
}

// #ifdef FPGADEBUG
// #pragma message "\n\n      FPGADEBUG is active \n\n"

// template <typename T>
// void printArray(std::vector<T, std::allocator<T>> data, int size, const char* name){
//     if (size >= 1) {
//         //printf("%s = [%d", name, data[0]);
//         std::cout << name << " = [" << data[0];
//         for (int i=1; i < size; i++){
//             //printf(", %d", data[i]);
//             std::cout << ", " << data[i];
//         }
//         //printf("]\n");
//         std::cout << "]\n";
//     } else {
//         //printf("%s []\n", name);
//         std::cout << name << " []\n";
//     }
    
// }
// template <typename T>
// void printArray(std::vector<T, aligned_allocator<T> > data, int size, const char* name){
//     if (size >= 1) {
//         //printf("%s = [%d", name, data[0]);
//         std::cout << name << " = [" << data[0];
//         for (int i=1; i < size; i++){
//             //printf(", %d", data[i]);
//             std::cout << ", " << data[i];
//         }
//         //printf("]\n");
//         std::cout << "]\n";
//     } else {
//         //printf("%s []\n", name);
//         std::cout << name << " []\n";
//     }
    
// }

// template <typename T>
// void printArray(T* data, int size, const char* name){
//     if (size >= 1) {
//         //printf("%s = [%d", name, data[0]);
//         std::cout << name << " = [" << data[0];
//         for (int i=1; i < size; i++){
//             //printf(", %d", data[i]);
//             std::cout << ", " << data[i];
//         }
//         //printf("]\n");
//         std::cout << "]\n";
//     } else {
//         //printf("%s []\n", name);
//         std::cout << name << " []\n";
//     }
    
// }

// template <typename T>
// void printArray(const T* data, int size, const char* name){
//     if (size >= 1) {
//         //printf("%s = [%d", name, data[0]);
//         std::cout << name << " = [" << data[0];
//         for (int i=1; i < size; i++){
//             //printf(", %d", data[i]);
//             std::cout << ", " << data[i];
//         }
//         //printf("]\n");
//         std::cout << "]\n";
//     } else {
//         //printf("%s []\n", name);
//         std::cout << name << " []\n";
//     }
    
// }
// #endif

// cl::Context context;
// cl::CommandQueue q;
// cl::Program program;

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
//int iteration = 0;

//bool device_setup = false;
// #ifdef FPGADEBUG
// bool dbg = true;
// #else
// bool dbg = false;
// #endif

 //cl::Program init_xilinx(cl::CommandQueue & q, cl::Context & context){
// void init_xilinx(cl::Context & context, cl::CommandQueue & q, cl::Program & program){
//     //mtx.lock(); //need to init FPGA only once

//     //cl::Program program;

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
//     //return program;
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

// #ifdef ONETHREAD
//     mtx.lock();
// #endif

//     iteration++;
// #ifdef FPGADEBUG
//     printf("\n\nFPGACALL call #%d==================\n\n", iteration);
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
//     // if (!device_setup){
//     //     printf("fpgasdacell: init device before fpgasdacell()\n");
//     //     precheck = false;
//     // }
//     if (!precheck){
//         printf("fgpasdacell: Wrong input parameters\n");
//         return -2;
//     }
//     if (work_size==0){
//         //printf("WORK_SIZE IS ZERO!!\n");
//         return 0;
//     }

// /////////////////////////////////////////////////////////////////////////////////////////////
//     init_xilinx(context, q, program);
//     // cl::Program program;

//     // //if (!device_setup){

//     // //XILINX FPGA SETUP
//     // // The get_xil_devices will return vector of Xilinx Devices 
//     // std::vector<cl::Device> devices = xcl::get_xil_devices();
//     // printf("Found %d devices\n", devices.size();
//     // cl::Device device = devices[0];

//     // //Creating Context and Command Queue for selected Device 
//     // cl::Context context(device);
//     // //context = cl::Context(device);
//     // cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);
//     // //q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
//     // std::string device_name = device.getInfo<CL_DEVICE_NAME>();

//     // // the OpenCL binary file created using the 
//     // // xocc compiler load into OpenCL Binary and return as Binaries
//     // // OpenCL and it can contain many functions which can be executed on the
//     // // device.
//     // std::string binaryFile = xcl::find_binary_file(device_name,"histogram");
//     // cl::Program::Binaries binary_file = xcl::import_binary_file(binaryFile);
//     // devices.resize(1);
//     // program = cl::Program(context, devices, binary_file);
//     #ifdef FPGADEBUG
//     printf("Program set\n");
//     #endif
// /////////////////////////////////////////////////////////////////////////////////////////////

//     //compute the size of array in bytes

//     // Creates a vector for data
//     //input
//     //data_pointer is a vector already. //
//     vector<int,aligned_allocator<int>> bins(data_size, 0);
//     vector<int,aligned_allocator<int>> indices(index_size, 0);
//     vector<float,aligned_allocator<float>> gradients(data_size, 0.0);
//     vector<float,aligned_allocator<float>> hessians(data_size, 0.0);
//     //output
//     vector<int,aligned_allocator<int>> histogram(histogram_size, 0);
//     vector<float,aligned_allocator<float>> sum_gradients(histogram_size, 0.0);
//     vector<float,aligned_allocator<float>> sum_hessians(histogram_size, 0.0);

//     //patches for unused arrays
//     vector<int,aligned_allocator<int>> intpatch(1, 0);
//     vector<float,aligned_allocator<float>> floatpatch(1, 0.0);

//     //device should be set up already
    
//     //fill data
//     //srand ( time(NULL) );
//     for (int i=0; i < data_size; i++){
//         int value = (int)data_pointer[i];
//         bins.insert(bins.begin() + i, value);

//         if (mode & HESSIANS) {
//             hessians.insert(hessians.begin() + i, hessian_pointer[i]);
//         }

//         if (mode & GRADIENTS) {
//             gradients.insert(gradients.begin() + i, gradient_pointer[i]);
//         }
//     }
//     if (mode & DATAINDICES) {
//         for (int i=0; i< index_size; i++){
//             int value = index_pointer[i];
//             indices.insert(indices.begin() + i, value);
//         }
//     }
//     for (int i=0; i < histogram_size; i++){
//         histogram.insert(histogram.begin() + i, histogram_couter[i]);

//         if (mode & HESSIANS) {
//             hessians.insert(hessians.begin() + i, histogram_hessian[i]);
//         }

//         if (mode & GRADIENTS) {
//             gradients.insert(gradients.begin() + i, gradient_pointer[i]);
//         }
//     }

// #ifdef FPGADEBUG
//         printf("=OLD_FPGGACALL_VERSION====\nINPUTS:\n");
//         //printArray<VAL_T>(bins, data_size, "data");
//         //printArray<VAL_T>(data_pointer, data_size, "data");
//         //printArray<int>(indices, index_size, "indices");
//         //printArray<float>(gradients, data_size, "gradients");
//         //printArray<float>(hessians, data_size, "hessians");
//         printArray<VAL_T>(data_pointer, min(data_size,10), "data");
//         printArray<int>(indices, min(index_size, 10), "indices");
//         printArray<float>(gradients, min(data_size, 10), "gradients");
//         printArray<float>(hessians, min(data_size, 10), "hessians");
//         VAL_T k;
//         printf("Size of VAL_T: %d and type of %s\n=====\n", sizeof(VAL_T), typeid(k).name());
//         unsigned char uc;
//         printf("Size of unsigned char: %d type %s\n", sizeof(unsigned char), typeid(uc).name());
//         unsigned short s;
//         printf("Size of unsigned short: %d and type of %s\n", sizeof(unsigned short), typeid(s).name());
//         unsigned int r;
//         printf("Size of unsigned int: %d and type of %s\n", sizeof(unsigned int), typeid(r).name());
//         printf("=====/inputs\n");
// #endif

//     // These commands will allocate memory on the FPGA
//     //input
//     cl::Buffer buffer_bins;
//     cl::Buffer buffer_inds;
//     cl::Buffer buffer_grad;
//     cl::Buffer buffer_hess;
//     // cl::Buffer buffer_bins(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, data_size_in_bytes, bins.data());

//     size_t data_size_in_bytes = data_pointer.size() * sizeof(int);
//     size_t grad_size_in_bytes = gradients.size() * sizeof(float);
//     size_t hess_size_in_bytes = hessians.size() * sizeof(float);
//     size_t index_size_in_bytes = index_size * sizeof(int);
//     size_t hist_size_in_bytes = histogram_size * sizeof(int);
//     size_t sums_size_in_bytes = histogram_size * sizeof(float);
    
//     //buffer_bins = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, data_size_in_bytes, const_cast<VAL_T*>(data_pointer.data()));
//     buffer_bins = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, data_size_in_bytes, bins.data());
//     if (mode & DATAINDICES) {
//         buffer_inds = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, index_size_in_bytes, indices.data());
//     } else {
//         buffer_inds = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 1, intpatch.data());
//     }
//     if (mode & GRADIENTS) {
//         buffer_grad = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, grad_size_in_bytes, gradients.data());
//     } else {
//         buffer_grad = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 1, floatpatch.data());
//     }
//     if (mode & HESSIANS) {
//         buffer_hess = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hess_size_in_bytes, hessians.data());
//     } else {
//         buffer_hess = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 1, floatpatch.data());
//     }
//     //output
//     cl::Buffer buffer_hist(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, hist_size_in_bytes, histogram.data());
//     cl::Buffer buffer_sgrad(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sums_size_in_bytes, sum_gradients.data());
//     cl::Buffer buffer_shess(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sums_size_in_bytes, sum_hessians.data());
    
//     //Separate Read/write Buffer vector is needed to migrate data between host/device
//     std::vector<cl::Memory> inBufVec, outBufVec;
//     inBufVec.push_back(buffer_bins);
//     inBufVec.push_back(buffer_inds);
//     inBufVec.push_back(buffer_grad);
//     inBufVec.push_back(buffer_hess);

//     outBufVec.push_back(buffer_hist);
//     outBufVec.push_back(buffer_sgrad);
//     outBufVec.push_back(buffer_shess);

//     // load data vector from the host to the FPGA
//     cl::Event inbuf_event;
//     cl_int err = q.enqueueMigrateMemObjects(inBufVec,0/* 0 means from host*/, NULL, &inbuf_event);
//     q.finish();
// #ifdef FPGADEBUG
//     uint64_t duration = get_duration_ns(inbuf_event);
//     printf("Argument load took %"PRIu64"\n", duration);
//     printf("MIGRATEMEMOBJECTS inBufVec (host -> device) code: %d\n", err);
// #endif
//     check(err, 781);

//     cl::Event outbuf_event;
//     err = q.enqueueMigrateMemObjects(outBufVec,0,NULL,&outbuf_event);
//     q.finish();
// #ifdef FPGADEBUG
//     duration = get_duration_ns(outbuf_event);
//     printf("Output preload took %"PRIu64"\n", duration);
//     printf("MIGRATEMEMOBJECTS inBufVec (host -> device) code: %d\n", err);
// #endif
//     check(err, 791);

// #ifdef FPGADEBUG
//         printf("MIGRATEMEMOBJECTS outBufVec (host -> device) code: %d\n", err);
// #endif

//     // extract a kernel
//     cl::Kernel krnl_hist_add(program, "hist_lightgbm");

//     //set the kernel Arguments
//     int narg=0;
//     int items_per_call = 1;
//     //int number_of_kernels = work_size / 64;
//     krnl_hist_add.setArg(narg++, items_per_call);//amount of data per one instance of a kernel
//     krnl_hist_add.setArg(narg++, work_size);//total size of work
//     krnl_hist_add.setArg(narg++, buffer_bins);//input bin indices
//     krnl_hist_add.setArg(narg++, buffer_inds);//input data indices
//     krnl_hist_add.setArg(narg++, buffer_grad);//input gradients
//     krnl_hist_add.setArg(narg++, buffer_hess);//input hessians
//     krnl_hist_add.setArg(narg++, buffer_sgrad);//output sum gradients
//     krnl_hist_add.setArg(narg++, buffer_shess);//output sum hessians
//     krnl_hist_add.setArg(narg++, buffer_hist);//output histogram (should not be NULL)
//     krnl_hist_add.setArg(narg++, mode);//sets what to use
// //#ifdef FPGADEBUG
// //    krnl_hist_add.setArg(narg++, 1);//do debug print
// //    printf("kernel print should work\n");
// //#else
//     krnl_hist_add.setArg(narg++, 0);//skip debug print
// //#endif

// #ifdef FPGADEBUG
//         printf("Kernel arguments should be set\n");
// #endif

//     //Launch the Kernel
//     cl::Event kernel_event;
//     cl_int krnl_err = q.enqueueNDRangeKernel(
//         krnl_hist_add, //kernel
//         {1},         //work_dim (offset)
//         {work_size}, //work_size (global)
//         {1},         //work_size (local)
//         NULL,
//         &kernel_event);
//     q.finish();

// #ifdef FPGADEBUG
//     duration = get_duration_ns(kernel_event);
//     printf("Kernel took %"PRIu64"\n", duration);
// #endif
//     check(krnl_err, 838);

//     // The result of the previous kernel execution will need to be retrieved in
//     // order to view the results. This call will write the data from the
//     // buffer_result cl_mem object to the source_results vector
//     err = q.enqueueMigrateMemObjects(outBufVec,CL_MIGRATE_MEM_OBJECT_HOST);
//     q.finish();
//     check(err, 845);

// #ifdef FPGADEBUG
//         printf("MIGRATEMEMOBJECTS outBufVec (host <- device) code: %d\n", err);
// #endif

//     q.finish();

// #ifdef FPGADEBUG
//         printf("====RESULTS\n");
//         printf("mode = %d\n", mode);
//         printArray<int>(histogram, min(histogram_size,10), "hist");
//         printArray<float>(sum_gradients, min(histogram_size,10), "sum_grad");
//         printArray<float>(sum_hessians, min(histogram_size,10), "sum_hess");
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

// #ifdef ONETHREAD
//     mtx.unlock();
// #endif
//     return err;
// }

using std::array;
using std::chrono::duration;
using std::chrono::nanoseconds;
using std::chrono::seconds;
using std::default_random_engine;
using std::generate;
using std::uniform_int_distribution;
using std::vector;

//Allocator template to align buffer to Page boundary for better data transfer
template <typename T>
struct aligned_allocator
{
  using value_type = T;
  T* allocate(std::size_t num)
  {
    void* ptr = nullptr;
    if (posix_memalign(&ptr,4096,num*sizeof(T)))
      throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
  }
  void deallocate(T* p, std::size_t num)
  {
    free(p);
  }
};

// Wrap any OpenCL API calls that return error code(cl_int)
// with the below macro to quickly check for an error
#define OCL_CHECK(call)                                              \
    do {                                                             \
        cl_int err = call;                                           \
        if (err != CL_SUCCESS) {                                     \
            printf("Error from " #call ", error code is %d\n", err); \
            exit(1);                                                 \
        }                                                            \
    } while (0);

// An event callback function that prints the operations performed by the OpenCL
// runtime.
void event_cb(cl_event event, cl_int cmd_status, void *data) {
  cl_command_type command;
  clGetEventInfo(event, CL_EVENT_COMMAND_TYPE, sizeof(cl_command_type),
                 &command, nullptr);
  cl_int status;
  clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int),
                 &status, nullptr);
  const char *command_str;
  const char *status_str;
  switch (command) {
  case CL_COMMAND_READ_BUFFER:
    command_str = "buffer read";
    break;
  case CL_COMMAND_WRITE_BUFFER:
    command_str = "buffer write";
    break;
  case CL_COMMAND_NDRANGE_KERNEL:
    command_str = "kernel";
    break;
  case CL_COMMAND_MAP_BUFFER:
    command_str = "kernel";
    break;
  case CL_COMMAND_COPY_BUFFER:
    command_str = "kernel";
    break;
  case CL_COMMAND_MIGRATE_MEM_OBJECTS:
        command_str = "buffer migrate";
      break;
  default:
    command_str = "unknown";
  }
  switch (status) {
  case CL_QUEUED:
    status_str = "Queued";
    break;
  case CL_SUBMITTED:
    status_str = "Submitted";
    break;
  case CL_RUNNING:
    status_str = "Executing";
    break;
  case CL_COMPLETE:
    status_str = "Completed";
    break;
  }
  printf("[%s]: %s %s\n", reinterpret_cast<char *>(data), status_str,
         command_str);
  fflush(stdout);
}

// Sets the callback for a particular event
void set_callback(cl_event event, const char *queue_name) {
  OCL_CHECK(
      clSetEventCallback(event, CL_COMPLETE, event_cb, (void *)queue_name));
}

int gen_empty_int(){
    return 0;
}
float gen_empty_float(){
    return 0;
}

int fpgacall(
             int    data_size,
             const std::vector<VAL_T, std::allocator<VAL_T>> data,
             const float* hessian_pointer,
             const float* gradient_pointer,
             int    index_size,
             const int*   index_pointer,
             int    histogram_size,
             float* histogram_hessian,
             float* histogram_gradient,
             int*   histogram_couter,
             int    mode){

    int DATA_SIZE = data_size;   //const 
    int ARRAY_SIZE = index_size;   //const 
    int HIST_SIZE = histogram_size;   //const 

    if (index_pointer == nullptr){
        ARRAY_SIZE = DATA_SIZE
    }

    cl_int err;
    xcl_world world = xcl_world_single();
    cl_program program = xcl_import_binary(world, "fpga64");

    size_t elements_per_iteration = ARRAY_SIZE;
    size_t bytes_per_iteration = elements_per_iteration * sizeof(int);
    size_t num_iterations = ARRAY_SIZE / elements_per_iteration;

    size_t elements_per_iteration_data = DATA_SIZE;
    size_t bytes_per_iteration_data = elements_per_iteration_data * sizeof(int);
    size_t num_iterations_data = DATA_SIZE / elements_per_iteration_data;

    size_t elements_per_iteration_float = ARRAY_SIZE;
    size_t bytes_per_iteration_float = elements_per_iteration_float * sizeof(float);
    size_t num_iterations_float = ARRAY_SIZE / elements_per_iteration_float;

    size_t elements_per_result = HIST_SIZE;
    size_t bytes_per_result_float = elements_per_result * sizeof(float);
    size_t bytes_per_result_int = elements_per_result * sizeof(int);
    //size_t num_iterations_float = ARRAY_SIZE / elements_per_iteration_float;

    //out of order command queue
    clReleaseCommandQueue(world.command_queue);
    world.command_queue =
      clCreateCommandQueue(world.context, world.device_id,
                           CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);

    //alloc memory and add data on host side
    //vector<int,aligned_allocator<int>> data(DATA_SIZE);
    vector<int,aligned_allocator<int>> index(ARRAY_SIZE);
    vector<float,aligned_allocator<float>> grad(ARRAY_SIZE);
    vector<float,aligned_allocator<float>> hess(ARRAY_SIZE);
    //generate(begin(data), end(data), gen_random_data);

    printf("Transforming input data to vectors\n");
    if (index_pointer == nullptr){
        //generate(begin(index), end(index), gen_random_int);
        for (int i=0; i < ARRAY_SIZE; i++){
            index.insert(index.begin() + i, i);
        }
    } else {
        for (int i=0; i < ARRAY_SIZE; i++){
            index.insert(index.begin() + i, index_pointer[i]);
        }
    }

    if (gradient_pointer == nullptr){
        generate(begin(grad), end(grad), gen_empty_float);
    } else {
        for (int i=0; i < ARRAY_SIZE; i++){
            grad.insert(grad.begin() + i, gradient_pointer[i]);
        }
    }

    if (hessian_pointer == nullptr){
        generate(begin(hess), end(hess), gen_empty_float);   
    } else {
        for (int i=0; i < ARRAY_SIZE; i++){
            hess.insert(hess.begin() + i, hessian_pointer[i]);
        }
    }

    vector<int,aligned_allocator<int>> result_count(HIST_SIZE);
    vector<float,aligned_allocator<float>> result_grad(HIST_SIZE);
    vector<float,aligned_allocator<float>> result_hess(HIST_SIZE);
    //generate(begin(result_count), end(result_count), gen_empty_int);
    //generate(begin(result_grad), end(result_grad), gen_empty_float);
    //generate(begin(result_hess), end(result_hess), gen_empty_float);

    for (int i=0; i < HIST_SIZE; i++){
        result_count.insert(result_count.begin() + i, histogram_couter[i]);
        result_grad.insert(result_grad.begin() + i, histogram_gradient[i]);
        result_hess.insert(result_hess.begin() + i, histogram_hessian[i]);
    }

    printf("Done\n");
    // for(int i = 0; i < 64; i++){
    //     printf("Host mem %d data %d, count %d\n",i,data[i],result_count[i]);
    // }

    //select kernel
    cl_kernel kernel = xcl_get_kernel(program, "kernel_fpgahistogram");

    //buffers for sending data
    array<cl_event, 2> kernel_events;
    array<cl_event, 6> read_events; //3 reads per iterations
    cl_mem buffer_data[2], buffer_idx[2], buffer_grad[2], buffer_hess[2], buffer_res_cnt[2], buffer_res_grad[2], buffer_res_hess[2];

    size_t global = 16, local = 16;
    size_t iteration_idx = 0;
    //for (size_t iteration_idx = 0; iteration_idx < num_iterations; iteration_idx++) {
    int flag = iteration_idx % 2;

    //not doing right now
    // if (iteration_idx >= 2) {
    //     clWaitForEvents(1, &read_events[flag]);
    //     OCL_CHECK(clReleaseMemObject(buffer_data[flag]));
    //     OCL_CHECK(clReleaseMemObject(buffer_idx[flag]));
    //     OCL_CHECK(clReleaseMemObject(buffer_grad[flag]));
    //     OCL_CHECK(clReleaseMemObject(buffer_hess[flag]));
    //     OCL_CHECK(clReleaseMemObject(buffer_res_cnt[flag]));
    //     OCL_CHECK(clReleaseMemObject(buffer_res_grad[flag]));
    //     OCL_CHECK(clReleaseMemObject(buffer_res_hess[flag]));
    //     OCL_CHECK(clReleaseEvent(read_events[flag]));
    //     OCL_CHECK(clReleaseEvent(kernel_events[flag]));
    // }

    buffer_data[flag] = clCreateBuffer(world.context,  
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
           bytes_per_iteration_data, &data[iteration_idx * elements_per_iteration_data], NULL);
    buffer_idx[flag] = clCreateBuffer(world.context,  
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
           bytes_per_iteration, &index[iteration_idx * elements_per_iteration], NULL);
    buffer_grad[flag] = clCreateBuffer(world.context,  
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
           bytes_per_iteration_float, &grad[iteration_idx * elements_per_iteration_float], NULL);
    buffer_hess[flag] = clCreateBuffer(world.context,  
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
           bytes_per_iteration_float, &hess[iteration_idx * elements_per_iteration_float], NULL);
    
    buffer_res_cnt[flag] = clCreateBuffer(world.context,  
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
           bytes_per_result_int, &result_count[iteration_idx * elements_per_result], NULL);
    buffer_res_grad[flag] = clCreateBuffer(world.context,  
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
           bytes_per_result_float, &result_grad[iteration_idx * elements_per_result], NULL);
    buffer_res_hess[flag] = clCreateBuffer(world.context,  
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
           bytes_per_result_float, &result_hess[iteration_idx * elements_per_result], NULL);

    array<cl_event, 7> write_events;
    printf("Enqueueing Migrate Mem Object (Host to Device) calls\n");

/*
cl_int clEnqueueMigrateMemObjects ( cl_command_queue  command_queue ,
    cl_uint  num_mem_objects ,
    const cl_mem  *mem_objects ,
    cl_mem_migration_flags  flags ,
    cl_uint  num_events_in_wait_list ,
    const cl_event  *event_wait_list ,
    cl_event  *event )
*/

    OCL_CHECK(clEnqueueMigrateMemObjects(
        world.command_queue, 1, &buffer_data[flag],
        0 /* flags, 0 means from host */,
        0, NULL, 
        &write_events[0]));
    set_callback(write_events[0], "ooo_queue_w_data");
    OCL_CHECK(clEnqueueMigrateMemObjects(
        world.command_queue, 1, &buffer_idx[flag],
        0 /* flags, 0 means from host */,
        0, NULL, 
        &write_events[1]));
    set_callback(write_events[1], "ooo_queue_w_idx");
    OCL_CHECK(clEnqueueMigrateMemObjects(
        world.command_queue, 1, &buffer_grad[flag],
        0 /* flags, 0 means from host */,
        0, NULL, 
        &write_events[2]));
    set_callback(write_events[2], "ooo_queue_w_grad");
    OCL_CHECK(clEnqueueMigrateMemObjects(
        world.command_queue, 1, &buffer_hess[flag],
        0 /* flags, 0 means from host */,
        0, NULL, 
        &write_events[3]));
    set_callback(write_events[3], "ooo_queue_w_hess");

    OCL_CHECK(clEnqueueMigrateMemObjects(
        world.command_queue, 1, &buffer_res_cnt[flag],
        0 /* flags, 0 means from host */,
        0, NULL, 
        &write_events[4]));
    set_callback(write_events[4], "ooo_queue_w_rcnt");
    OCL_CHECK(clEnqueueMigrateMemObjects(
        world.command_queue, 1, &buffer_res_grad[flag],
        0 /* flags, 0 means from host */,
        0, NULL, 
        &write_events[5]));
    set_callback(write_events[5], "ooo_queue_w_rgrad");
    OCL_CHECK(clEnqueueMigrateMemObjects(
        world.command_queue, 1, &buffer_res_hess[flag],
        0 /* flags, 0 means from host */,
        0, NULL, 
        &write_events[6]));
    set_callback(write_events[6], "ooo_queue_w_rhess");

/*
clGetDeviceInfo(    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret)
*/
    //ulong local_size = 0;
    //OCL_CHECK(clGetDeviceInfo(world.device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_size, NULL));
    //printf("Local memory size is: %lu bytes\n", local_size);


    int amount_of_work = ARRAY_SIZE;
    int use = 1;
    
    //          const int num_data, //amount of work              0
    xcl_set_kernel_arg(kernel, 0, sizeof(int), &amount_of_work);
    //          const int use_index,//flag for index calculation  1
    xcl_set_kernel_arg(kernel, 1, sizeof(int), &use);
    //          const int use_hess, //flag for hessian            2
    xcl_set_kernel_arg(kernel, 2, sizeof(int), &use);
    // __global const int* data,    //data pointer                3
    xcl_set_kernel_arg(kernel, 3, sizeof(cl_mem), &buffer_data[iteration_idx % 2]);
    // __global const int* index,   //index pointer (might be 0)  4
    xcl_set_kernel_arg(kernel, 4, sizeof(cl_mem), &buffer_idx[iteration_idx % 2]);
    // __global const float* grad,  //gradient input pointer      5
    xcl_set_kernel_arg(kernel, 5, sizeof(cl_mem), &buffer_grad[iteration_idx % 2]);
    // __global const float* hess,  //hessian input pointer       6
    xcl_set_kernel_arg(kernel, 6, sizeof(cl_mem), &buffer_hess[iteration_idx % 2]);
    // __global       int* count,   //counter output              7
    xcl_set_kernel_arg(kernel, 7, sizeof(cl_mem), &buffer_res_cnt[iteration_idx % 2]);
    // __global       float* hgrad, //gradient output             8
    xcl_set_kernel_arg(kernel, 8, sizeof(cl_mem), &buffer_res_grad[iteration_idx % 2]);
    // __global       float* hhess, //hessian output              9
    xcl_set_kernel_arg(kernel, 9, sizeof(cl_mem), &buffer_res_hess[iteration_idx % 2]);

    // __local        int*  Lhist,  //NUM_BINS*loc_size          10
    xcl_set_kernel_arg(kernel, 10, 64*16*sizeof(int), NULL);
    // __local        float* Lgrad, //NUM_BINS*loc_size          11
    xcl_set_kernel_arg(kernel, 11, 64*16*sizeof(float), NULL);
    // __local        float* Lhess, //NUM_BINS*loc_size          12
    xcl_set_kernel_arg(kernel, 12, 64*16*sizeof(float), NULL);
    // __local        int*  Lidx    //numdata                    13
    xcl_set_kernel_arg(kernel, 13, amount_of_work*sizeof(int), NULL);

    printf("Enqueueing NDRange kernel.\n");

/*
cl_int clEnqueueNDRangeKernel ( cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t *global_work_offset,
    const size_t *global_work_size,
    const size_t *local_work_size,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event)
*/

    OCL_CHECK(clEnqueueNDRangeKernel(world.command_queue,
                                     kernel, 1, nullptr,
                                     &global, &local, 7, write_events.data(),
                                     &kernel_events[flag]));
    set_callback(kernel_events[flag], "ooo_queue_kernel");

    printf("Enqueueing Migrate Mem Object (Device to Host) calls\n");
    // This operation only needs to wait for the kernel call. This call will
    // potentially overlap the next kernel call as well as the next read
    // operations

/*
cl_int clEnqueueMigrateMemObjects ( cl_command_queue  command_queue ,
    cl_uint  num_mem_objects ,
    const cl_mem  *mem_objects ,
    cl_mem_migration_flags  flags ,
    cl_uint  num_events_in_wait_list ,
    const cl_event  *event_wait_list ,
    cl_event  *event )
*/

    OCL_CHECK(clEnqueueMigrateMemObjects(world.command_queue, 1, &buffer_res_cnt[flag], 
                CL_MIGRATE_MEM_OBJECT_HOST, 1, &kernel_events[flag], &read_events[3 * flag]));
    OCL_CHECK(clEnqueueMigrateMemObjects(world.command_queue, 1, &buffer_res_grad[flag], 
                CL_MIGRATE_MEM_OBJECT_HOST, 1, &kernel_events[flag], &read_events[1 + 3 * flag]));
    OCL_CHECK(clEnqueueMigrateMemObjects(world.command_queue, 1, &buffer_res_hess[flag], 
                CL_MIGRATE_MEM_OBJECT_HOST, 1, &kernel_events[flag], &read_events[2 + 3 * flag]));

    set_callback(read_events[3 * flag], "ooo_queue_r_rcnt");
    set_callback(read_events[1 + 3 * flag], "ooo_queue_r_rgrad");
    set_callback(read_events[2 + 3 * flag], "ooo_queue_r_rhess");

    OCL_CHECK(clReleaseEvent(write_events[0]));
    OCL_CHECK(clReleaseEvent(write_events[1]));
    OCL_CHECK(clReleaseEvent(write_events[2]));
    OCL_CHECK(clReleaseEvent(write_events[3]));
    OCL_CHECK(clReleaseEvent(write_events[4]));
    OCL_CHECK(clReleaseEvent(write_events[5]));
    OCL_CHECK(clReleaseEvent(write_events[6]));

    //} //end of for loop

    printf("Waiting...\n");
    clFlush(world.command_queue);
    clFinish(world.command_queue);

    //Releasing mem objects and events
    //for(int i = 0 ; i < 2 ; i++){
    int i = 0;
        printf("releasing buffers i=%d\n",i);
        OCL_CHECK(clWaitForEvents(1, &read_events[3*i]));
        OCL_CHECK(clWaitForEvents(1, &read_events[3*i+1]));
        OCL_CHECK(clWaitForEvents(1, &read_events[3*i+2]));
        OCL_CHECK(clReleaseMemObject(buffer_data[i]));
        OCL_CHECK(clReleaseMemObject(buffer_idx[i]));
        OCL_CHECK(clReleaseMemObject(buffer_grad[i]));
        OCL_CHECK(clReleaseMemObject(buffer_hess[i]));
        OCL_CHECK(clReleaseMemObject(buffer_res_cnt[i]));
        OCL_CHECK(clReleaseMemObject(buffer_res_hess[i]));
        OCL_CHECK(clReleaseMemObject(buffer_res_grad[i]));
        OCL_CHECK(clReleaseEvent(read_events[3*i]));
        OCL_CHECK(clReleaseEvent(read_events[3*i+1]));
        OCL_CHECK(clReleaseEvent(read_events[3*i+2]));
        OCL_CHECK(clReleaseEvent(kernel_events[i]));
    //}

    // int match = 0;
    // // verify the results
    // int host_count[HIST_SIZE];
    // float host_grad[HIST_SIZE];
    // float host_hess[HIST_SIZE];
    // for (int i= 0; i < HIST_SIZE; i++){
    //     host_count[i] = 0;
    //     host_grad[i] = 0;
    //     host_hess[i] = 0;
    // }
    // //int bin_met = 0;
    // for (int i = 0; i < ARRAY_SIZE; i++){
    //     int bin = data[index[i]];
    //     if (bin == 49){
    //     //    bin_met++;
    //         printf("HOST CALC 49 old %f add grad[%d] %f\n", host_grad[bin], i, grad[i]);
    //     }
    //     //printf("idx %d counts=[%d,%d,%d,%d,%d,%d,%d,%d]\n", i, host_count[0], host_count[1], host_count[2], host_count[3], host_count[4], host_count[5], host_count[6], host_count[7]);
    //     host_count[bin]++;
    //     host_hess[bin] += hess[i];
    //     host_grad[bin] += grad[i];
    //     if (bin == 49){
    //         printf("HOST CALC 49 result %f\n", host_grad[bin]);
    //     }
    // }
    // printf("HOST grad 49 = %f\n", host_grad[49]);
    // //printf("Averall bin=0 met %i times, host_count[0] = %d\n", bin_met, host_count[0]);
    // int count_mis=0, grad_mis=0, hess_mis=0;
    // for (int i = 0; i < HIST_SIZE; i++) {
    //     int host_result = host_count[i];
    //     if (host_result != result_count[i]) {
    //         printf("mismatch of count at %d: CPU: %d, FPGA: %d, diff %d\n", i, host_result, result_count[i], std::abs(host_result - result_count[i]));
    //         match = 1;
    //         count_mis++;
    //     }
    //     float host_result_f = host_grad[i];
    //     if (host_result_f != result_grad[i]) {
    //         printf("mismatch of gradient at %d: CPU: %f, FPGA: %f, diff %f\n", i, host_result_f, result_grad[i], std::abs(host_result_f - result_grad[i]));
    //         match = 1;
    //         grad_mis++;
    //     }
    //     host_result_f = host_hess[i];
    //     if (host_result_f != result_hess[i]) {
    //         printf("mismatch of hessian at %d: CPU: %f, FPGA: %f, diff %f\n", i, host_result_f, result_hess[i], std::abs(host_result_f - result_hess[i]));
    //         match = 1;
    //         hess_mis++;
    //     }
    // }


    // printf("TEST %s\n", (match ? "FAILED" : "PASSED"));
    // printf("CNT: %d, GRD: %d, HES: %d\n", count_mis, grad_mis, hess_mis);

    OCL_CHECK(clReleaseKernel(kernel));
    OCL_CHECK(clReleaseProgram(program));
    xcl_release_world(world);

    //move out data
    for (int i=0; i < HIST_SIZE; i++){
        histogram_couter[i] = result_count[i];
        if (mode & GRADIENTS){
            histogram_gradient[i] = result_grad[i];
        }
        if (mode & HESSIANS){
            histogram_hessian[i] = result_hess[i];
        }
    }


}



//#endif //USE_OLD_FPGACALL

#endif //FPGASDACELL
