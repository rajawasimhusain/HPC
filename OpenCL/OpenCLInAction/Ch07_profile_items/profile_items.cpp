#include "../Common/helper.h"

#define PROGRAM_FILE "profile_items.cl"
#define KERNEL_FUNC "profile_items"

#define NUM_INTS 4096
#define NUM_ITEMS 512
#define NUM_ITERATIONS 2000

int main() 
{
   /* OpenCL data structures */
   cl_device_id device;
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel kernel;
   size_t num_items;
   cl_int err, num_ints;

   /* Data and events */
   int data[NUM_INTS];
   cl_mem data_buffer;
   cl_event prof_event;
   cl_ulong time_start, time_end, total_time;

   /* Initialize data */
   for(cl_int i=0; i<NUM_INTS; i++) {
      data[i] = i;
   }

   /* Set number of data points and work-items */
   num_ints = NUM_INTS;
   num_items = NUM_ITEMS;

   /* Create a device and context */
   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

   /* Build the program and create a kernel */
   program = build_program(context, device, PROGRAM_FILE);
   kernel = clCreateKernel(program, KERNEL_FUNC, &err);

   /* Create a buffer to hold data */
   data_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(data), data, &err);

   /* Create kernel argument */
   clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_buffer);
   clSetKernelArg(kernel, 1, sizeof(num_ints), &num_ints);

   /* Create a command queue */
   queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

   total_time = 0;
   for(cl_int i=0; i<NUM_ITERATIONS; i++)
	 {         
      /* Enqueue kernel */
      clEnqueueNDRangeKernel(queue,       // queue
				                     kernel,      // kernel
				                     1,           // work_dims 
				                     NULL,        // *global_work_offset
				                     &num_items,  // *global_work_size  
				                     NULL,        // *local_work_size
				                     0,           // num_events
				                     NULL,        // *wait_list
				                     &prof_event);// *event

      /* Finish processing the queue and get profiling information */
      clFinish(queue);
      clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START,  sizeof(time_start), &time_start, NULL);
      clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
      total_time += time_end - time_start;
   }
   printf("Average time = %lu ns\n", total_time/NUM_ITERATIONS);

	 getchar();

   /* Deallocate resources */
   clReleaseEvent(prof_event);
   clReleaseKernel(kernel);
   clReleaseMemObject(data_buffer);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}
