#include "../Common/helper.h"

#define PROGRAM_FILE "profile_read.cl"
#define KERNEL_FUNC "profile_read"

#define NUM_BYTES 131072
#define NUM_ITERATIONS 2000
//#define PROFILE_READ 1

int main() 
{
   /* OpenCL data structures */
   cl_device_id device;
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel kernel;
   cl_int i, err, num_vectors;

   /* Data and events */
   char data[NUM_BYTES];
   cl_mem data_buffer;
   cl_event prof_event;
   cl_ulong time_start, time_end, total_time;
   void* mapped_memory;

   /* Create a device and context */
   device  = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

   /* Build the program and create a kernel */
   program = build_program(context, device, PROGRAM_FILE);
   kernel  = clCreateKernel(program, KERNEL_FUNC, &err);

   /* Create a buffer to hold data */
   data_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(data), NULL, &err);

   /* Create kernel argument */
   clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_buffer);

   /* Tell kernel number of char16 vectors */
   num_vectors = NUM_BYTES/16;
   clSetKernelArg(kernel, 1, sizeof(num_vectors), &num_vectors);

   /* Create a command queue */
   queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

   total_time = 0;
   for(i=0; i<NUM_ITERATIONS; i++) 
	 {
      /* Enqueue kernel */
      clEnqueueTask(queue, kernel, 0, NULL, NULL);

#ifdef PROFILE_READ
      /* Read the buffer */
      clEnqueueReadBuffer(queue, data_buffer, CL_TRUE, 0, sizeof(data), data, 0, NULL, &prof_event);

      /* Get profiling information for the read */
      clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
      clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
      total_time += time_end - time_start;

#else
      /* Create memory map */
      mapped_memory = clEnqueueMapBuffer(queue, data_buffer, CL_TRUE, CL_MAP_READ, 0, sizeof(data), 0, NULL, &prof_event, &err);

      /* Get profiling information for the map */
      clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
      clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
      total_time += time_end - time_start;

      /* Unmap the buffer */
      clEnqueueUnmapMemObject(queue, data_buffer, mapped_memory, 0, NULL, NULL);
#endif
   }

#ifdef PROFILE_READ
   printf("Average read time: %lu ns\n", total_time/NUM_ITERATIONS);
#else
   printf("Average map time: %lu ns\n", total_time/NUM_ITERATIONS);
#endif

	 getchar();
   /* Deallocate resources */
   clReleaseEvent(prof_event);
   clReleaseMemObject(data_buffer);
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}
