// Copyright (C) 2013-2014 Altera Corporation, San Jose, California, USA. All rights reserved. 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this 
// software and associated documentation files (the "Software"), to deal in the Software 
// without restriction, including without limitation the rights to use, copy, modify, merge, 
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to 
// whom the Software is furnished to do so, subject to the following conditions: 
// The above copyright notice and this permission notice shall be included in all copies or 
// substantial portions of the Software. 
//  
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
// OTHER DEALINGS IN THE SOFTWARE. 
//  
// This agreement shall be governed in all respects by the laws of the State of California and 
// by the laws of the United States of America. 

///////////////////////////////////////////////////////////////////////////////////
// This host program executes a vector addition kernel to perform:
//  C = A + B
// where A, B and C are vectors with N elements.
//
// This host program supports partitioning the problem across multiple OpenCL
// devices if available. If there are M available devices, the problem is
// divided so that each device operates on N/M points. The host program
// assumes that all devices are of the same type (that is, the same binary can
// be used), but the code can be generalized to support different device types
// easily.
//
// Verification is performed against the same computation on the host CPU.
///////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/opencl.h"
#include "AOCL_Utils.h"

using namespace aocl_utils;

#define STRING_BUFFER_LEN 1024
const char *cl_kernel_file="vectorAdd";


// OpenCL runtime configuration
cl_platform_id platform = NULL;
cl_uint num_devices = 0;
scoped_array<cl_device_id> device; // num_devices elements
cl_context context = NULL;
scoped_array<cl_command_queue> queue; // num_devices elements
cl_program program = NULL;
scoped_array<cl_kernel> kernel; // num_devices elements
scoped_array<cl_mem> input_a_buf; // num_devices elements
scoped_array<cl_mem> input_b_buf; // num_devices elements
scoped_array<cl_mem> output_buf; // num_devices elements

// Problem data.
const unsigned N = 1000000; // problem size
scoped_array<scoped_aligned_ptr<float> > input_a, input_b; // num_devices elements
scoped_array<scoped_aligned_ptr<float> > output; // num_devices elements
scoped_array<scoped_array<float> > ref_output; // num_devices elements
scoped_array<unsigned> n_per_device; // num_devices elements

// Function prototypes
float rand_float();
bool init_opencl();
void init_problem();
void run();
const char* GetDeviceType(cl_device_type it);
void cleanup();

// Entry point.
int main() {
  // Initialize OpenCL.
  if(!init_opencl()) {
    return -1;
  }

  // Initialize the problem data.
  // Requires the number of devices to be known.
  init_problem();

  // Run the kernel.
  run();

  // Free the resources allocated
  cleanup();

  return 0;
}

/////// HELPER FUNCTIONS ///////

// Randomly generate a floating-point number between -10 and 10.
float rand_float() {
  return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}

void ShowBuildLog() {
	// ****************Shows build log********************************
	char* build_log;
	size_t log_size;
	// First call to know the proper size
	clGetProgramBuildInfo(program, device[0], CL_PROGRAM_BUILD_LOG, 0, NULL,
			&log_size);
	build_log = new char[log_size + 1];
	//memset(build_log,0x00,(log_size + 1)*sizeof(char));
	// Second call to get the log
	clGetProgramBuildInfo(program, device[0], CL_PROGRAM_BUILD_LOG, log_size,
			build_log, NULL);
	build_log[log_size] = '\0';
	printf("build_log total [%d] chars : %s \n", strlen(build_log), build_log);
	delete[] build_log;
}

// Initializes the OpenCL objects.
bool init_opencl() {
  cl_int status;

  printf("Initializing OpenCL\n");


  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Altera");
  if(platform == NULL) {
    printf("ERROR: Unable to find Altera OpenCL platform.\n");
    return false;
  }

  // User-visible output - Platform information
  {
     char char_buffer[STRING_BUFFER_LEN];
     printf("Querying platform for info:\n");
     printf("==========================\n");
     clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
     printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
     clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
     printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
     clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
     printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);
  }

  // Query the available OpenCL device.
	  device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
	  printf("Platform: %s\n", getPlatformName(platform).c_str());
	  printf("found %d device(s)on the platform %s \n", num_devices,getPlatformName(platform).c_str());
	  printf("Querying device(s) for info:\n");
	  printf("==========================\n");
  for(unsigned i = 0; i < num_devices; ++i)
  {
	char char_buffer[STRING_BUFFER_LEN];
	cl_device_type int_type;
	cl_ulong long_entries;
	size_t MaxItemSize[3] = {0};
    //printf("deviceId[%d],device name[ %s ] \n",i, getDeviceName(device[i]).c_str());
	printf("deviceId[%d]\n",i);
	//获取设备名称
	clGetDeviceInfo(device[i], CL_DEVICE_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
	printf("%-40s = %s\n","CL_DEVICE_NAME",char_buffer);
	//获取设备类别
	clGetDeviceInfo(device[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &int_type, NULL);
	printf("%-40s = %s\n","CL_DEVICE_TYPE",GetDeviceType(int_type));

	//获取设备版本号
	clGetDeviceInfo(device[i], CL_DRIVER_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
	printf("%-40s = %s\n","CL_DRIVER_VERSION",char_buffer);

	//获取设备全局内存大小
	clGetDeviceInfo(device[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &long_entries, NULL);
	printf("%-40s = %d\n","CL_DEVICE_GLOBAL_MEM_SIZE(MB)",long_entries/1024/1024);
	//获取设备CACHE内存大小
	clGetDeviceInfo(device[i], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &long_entries, NULL);
	printf("%-40s = %d\n","CL_DEVICE_GLOBAL_MEM_CACHE_SIZE(KB)",long_entries/1024);

	//获取本地内存大小
	clGetDeviceInfo(device[i], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &long_entries, NULL);
	printf("%-40s = %d\n","CL_DEVICE_LOCAL_MEM_SIZE(KB)",long_entries/1024);

	//获取设备频率
	clGetDeviceInfo(device[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_ulong), &long_entries, NULL);
	printf("%-40s = %d\n","CL_DEVICE_MAX_CLOCK_FREQUENCY(MHz)",long_entries);

	//获取最大工作组数
	clGetDeviceInfo(device[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(cl_ulong), &long_entries, NULL);
	printf("%-40s = %d\n","CL_DEVICE_MAX_WORK_GROUP_SIZE",long_entries);
	//获取每个维度work-item size
	clGetDeviceInfo(device[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(cl_ulong), &long_entries, NULL);
	printf("%-40s = %d\n","CL_DEVICE_MAX_WORK_ITEM_SIZES",long_entries);

	//获取最大计算核心数
	clGetDeviceInfo(device[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_ulong), &long_entries, NULL);
	printf("%-40s = %d\n","CL_DEVICE_MAX_COMPUTE_UNITS",long_entries);

	//判断是否支持图像
	clGetDeviceInfo(device[i], CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_ulong), &long_entries, NULL);
	printf("%-40s = %s\n","CL_DEVICE_IMAGE_SUPPORT",long_entries==1?"true":"false");
	//最大采样器数目
	clGetDeviceInfo(device[i], CL_DEVICE_MAX_SAMPLERS, sizeof(cl_ulong), &long_entries, NULL);
	printf("%-40s = %d\n","CL_DEVICE_MAX_SAMPLERS",long_entries);
	printf("\n");
  }

  // Create the context.
  context = clCreateContext(NULL, num_devices, device, NULL, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the program for all device. Use the first device as the
  // representative device (assuming all device are of the same type).
  //const char *cl_kernel_file="vectorAdd";
  std::string binary_file = getBoardBinaryFile(cl_kernel_file, device[0]);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), device, num_devices);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // ****************Shows build log********************************
	ShowBuildLog();
  // ****************Shows build log end********************************

  // Create per-device objects.
  queue.reset(num_devices);
  kernel.reset(num_devices);
  n_per_device.reset(num_devices);
  input_a_buf.reset(num_devices);
  input_b_buf.reset(num_devices);
  output_buf.reset(num_devices);

  for(unsigned i = 0; i < num_devices; ++i) {
    // Command queue.
	printf("create CommandQueue & Kernel & Buffer(memery object) for device %d ... \r\n",i);
    queue[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    // Kernel.
    const char *kernel_name = "vectorAdd";
    kernel[i] = clCreateKernel(program, kernel_name, &status);
    checkError(status, "Failed to create kernel");

    // Determine the number of elements processed by this device.
    n_per_device[i] = N / num_devices; // number of elements handled by this device

    // Spread out the remainder of the elements over the first
    // N % num_devices.
    if(i < (N % num_devices)) {
      n_per_device[i]++;
    }

    // Input buffers.
    input_a_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, 
        n_per_device[i] * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input A");

    input_b_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, 
        n_per_device[i] * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input B");

    // Output buffer.
    output_buf[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
        n_per_device[i] * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");
  }

  return true;
}

// Initialize the data for the problem. Requires num_devices to be known.
void init_problem()
{
  if(num_devices == 0)
  {
    checkError(-1, "No devices");
  }

  input_a.reset(num_devices);
  input_b.reset(num_devices);
  output.reset(num_devices);
  ref_output.reset(num_devices);

  // Generate input vectors A and B and the reference output consisting
  // of a total of N elements.
  // We create separate arrays for each device so that each device has an
  // aligned buffer. 
  for(unsigned i = 0; i < num_devices; ++i)
  {
    input_a[i].reset(n_per_device[i]);
    input_b[i].reset(n_per_device[i]);
    output[i].reset(n_per_device[i]);
    ref_output[i].reset(n_per_device[i]);

    for(unsigned j = 0; j < n_per_device[i]; ++j)
    {
      input_a[i][j] = rand_float();
      input_b[i][j] = rand_float();
      ref_output[i][j] = input_a[i][j] + input_b[i][j];
    }
  }
}

void run() {
  cl_int status;

  const double start_time = getCurrentTimestamp();

  // Launch the problem for each device.
  scoped_array<cl_event> kernel_event(num_devices);
  scoped_array<cl_event> finish_event(num_devices);

  for(unsigned i = 0; i < num_devices; ++i) {

    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    cl_event write_event[2];
    status = clEnqueueWriteBuffer(queue[i], input_a_buf[i], CL_FALSE,
        0, n_per_device[i] * sizeof(float), input_a[i], 0, NULL, &write_event[0]);
    checkError(status, "Failed to transfer input A");

    status = clEnqueueWriteBuffer(queue[i], input_b_buf[i], CL_FALSE,
        0, n_per_device[i] * sizeof(float), input_b[i], 0, NULL, &write_event[1]);
    checkError(status, "Failed to transfer input B");

    // Set kernel arguments.
    unsigned argi = 0;

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &input_a_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &input_b_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &output_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);

    // Enqueue kernel.
    // Use a global work size corresponding to the number of elements to add
    // for this device.
    // 
    // We don't specify a local work size and let the runtime choose
    // (it'll choose to use one work-group with the same size as the global
    // work-size).
    //
    // Events are used to ensure that the kernel is not launched until
    // the writes to the input buffers have completed.
    const size_t global_work_size = n_per_device[i];
    printf("Launching for device %d (%d elements)\n", i, global_work_size);

    status = clEnqueueNDRangeKernel(queue[i], kernel[i], 1, NULL,
        &global_work_size, NULL, 2, write_event, &kernel_event[i]);
    checkError(status, "Failed to launch kernel");

    // Read the result. This the final operation.
    status = clEnqueueReadBuffer(queue[i], output_buf[i], CL_FALSE,
        0, n_per_device[i] * sizeof(float), output[i], 1, &kernel_event[i], &finish_event[i]);


    // Release local events.
    clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);
  }

  // Wait for all devices to finish.
  clWaitForEvents(num_devices, finish_event);

  const double end_time = getCurrentTimestamp();

  // Wall-clock time taken.
  printf("\ncost Time: %0.3f ms\n", (end_time - start_time) * 1e3);

  // Get kernel times using the OpenCL event profiling API.
  for(unsigned i = 0; i < num_devices; ++i) {
    cl_ulong time_ns = getStartEndTime(kernel_event[i]);
    printf("Kernel time (device %d): %0.3f ms\n", i, double(time_ns) * 1e-6);
  }

  // Release all events.
  for(unsigned i = 0; i < num_devices; ++i) {
    clReleaseEvent(kernel_event[i]);
    clReleaseEvent(finish_event[i]);
  }

  // Verify results.
  bool pass = true;
  for(unsigned i = 0; i < num_devices && pass; ++i) {
    for(unsigned j = 0; j < n_per_device[i] && pass; ++j) {
      if(fabsf(output[i][j] - ref_output[i][j]) > 1.0e-5f) {
        printf("Failed verification @ device %d, index %d\nOutput: %f\nReference: %f\n",
            i, j, output[i][j], ref_output[i][j]);
        pass = false;
      }
    }
  }

  printf("\nVerification: %s\n", pass ? "PASS" : "FAIL");
}

//根据参数，判断设备类别。是CPU、GPU、ACCELERATOR或其他设备
const char* GetDeviceType(cl_device_type it)
{
	if (it == CL_DEVICE_TYPE_CPU)
		return "CPU";
	else if (it == CL_DEVICE_TYPE_GPU)
		return "GPU";
	else if (it == CL_DEVICE_TYPE_ACCELERATOR)
		return "ACCELERATOR(FPGA)";
	else
		return "DEFAULT";

}

// Free the resources allocated during initialization
void cleanup() {
  for(unsigned i = 0; i < num_devices; ++i) {
    if(kernel && kernel[i]) {
      clReleaseKernel(kernel[i]);
    }
    if(queue && queue[i]) {
      clReleaseCommandQueue(queue[i]);
    }
    if(input_a_buf && input_a_buf[i]) {
      clReleaseMemObject(input_a_buf[i]);
    }
    if(input_b_buf && input_b_buf[i]) {
      clReleaseMemObject(input_b_buf[i]);
    }
    if(output_buf && output_buf[i]) {
      clReleaseMemObject(output_buf[i]);
    }
  }

  if(program) {
    clReleaseProgram(program);
  }
  if(context) {
    clReleaseContext(context);
  }
}

