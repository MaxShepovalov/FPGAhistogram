//host test

#include "CL/cl.h"
#include "xcl.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include <cmath>

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

const int DATA_SIZE = 1 << 14;
const int ARRAY_SIZE = 1024;
const int HIST_SIZE = 64;

int gen_random_data() {
    static default_random_engine e;
    static uniform_int_distribution<int> dist(0, HIST_SIZE-1);
    return dist(e);
}
int gen_random_int() {
    static default_random_engine e;
    static uniform_int_distribution<int> dist(0, DATA_SIZE-1);
    return dist(e);
}
float gen_random_float() {
    static default_random_engine e;
    static uniform_int_distribution<int> dist(0, 100);
    return dist(e) / 100.0;
}
int gen_empty_int(){
    return 0;
}
float gen_empty_float(){
    return 0;
}

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

int main(int argc, char **argv) {
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
    vector<int,aligned_allocator<int>> data(DATA_SIZE);
    vector<int,aligned_allocator<int>> index(ARRAY_SIZE);
    vector<float,aligned_allocator<float>> grad(ARRAY_SIZE);
    vector<float,aligned_allocator<float>> hess(ARRAY_SIZE);
    generate(begin(data), end(data), gen_random_data);
    generate(begin(index), end(index), gen_random_int);
    generate(begin(grad), end(grad), gen_random_float);
    generate(begin(hess), end(hess), gen_random_float);

    vector<int,aligned_allocator<int>> result_count(HIST_SIZE);
    vector<float,aligned_allocator<float>> result_grad(HIST_SIZE);
    vector<float,aligned_allocator<float>> result_hess(HIST_SIZE);
    generate(begin(result_count), end(result_count), gen_empty_int);
    generate(begin(result_grad), end(result_grad), gen_empty_float);
    generate(begin(result_hess), end(result_hess), gen_empty_float);

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

    int match = 0;
    // verify the results
    int host_count[HIST_SIZE];
    float host_grad[HIST_SIZE];
    float host_hess[HIST_SIZE];
    for (int i= 0; i < HIST_SIZE; i++){
        host_count[i] = 0;
        host_grad[i] = 0;
        host_hess[i] = 0;
    }
    //int bin_met = 0;
    for (int i = 0; i < ARRAY_SIZE; i++){
        int bin = data[index[i]];
        if (bin == 49){
        //    bin_met++;
            printf("HOST CALC 49 old %f add grad[%d] %f\n", host_grad[bin], i, grad[i]);
        }
        //printf("idx %d counts=[%d,%d,%d,%d,%d,%d,%d,%d]\n", i, host_count[0], host_count[1], host_count[2], host_count[3], host_count[4], host_count[5], host_count[6], host_count[7]);
        host_count[bin]++;
        host_hess[bin] += hess[i];
        host_grad[bin] += grad[i];
        if (bin == 49){
            printf("HOST CALC 49 result %f\n", host_grad[bin]);
        }
    }
    printf("HOST grad 49 = %f\n", host_grad[49]);
    //printf("Averall bin=0 met %i times, host_count[0] = %d\n", bin_met, host_count[0]);
    int count_mis=0, grad_mis=0, hess_mis=0;
    for (int i = 0; i < HIST_SIZE; i++) {
        int host_result = host_count[i];
        if (host_result != result_count[i]) {
            printf("mismatch of count at %d: CPU: %d, FPGA: %d, diff %d\n", i, host_result, result_count[i], std::abs(host_result - result_count[i]));
            match = 1;
            count_mis++;
        }
        float host_result_f = host_grad[i];
        if (host_result_f != result_grad[i]) {
            printf("mismatch of gradient at %d: CPU: %f, FPGA: %f, diff %f\n", i, host_result_f, result_grad[i], std::abs(host_result_f - result_grad[i]));
            match = 1;
            grad_mis++;
        }
        host_result_f = host_hess[i];
        if (host_result_f != result_hess[i]) {
            printf("mismatch of hessian at %d: CPU: %f, FPGA: %f, diff %f\n", i, host_result_f, result_hess[i], std::abs(host_result_f - result_hess[i]));
            match = 1;
            hess_mis++;
        }
    }

    OCL_CHECK(clReleaseKernel(kernel));
    OCL_CHECK(clReleaseProgram(program));
    xcl_release_world(world);

    printf("TEST %s\n", (match ? "FAILED" : "PASSED"));
    printf("CNT: %d, GRD: %d, HES: %d\n", count_mis, grad_mis, hess_mis);
    return (match ? EXIT_FAILURE :  EXIT_SUCCESS);
}
