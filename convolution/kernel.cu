#include "cuda_runtime.h"
#include "CImg.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace cimg_library;

cudaError_t convImage(CImg<unsigned char> *img);

__global__ void convolution(float *d_img, float *d_img_conv, int width, int height) {
	// Naive implementation
	// global mem address for this thread
	const int j = blockIdx.x * blockDim.x + threadIdx.x; // width
	const int i = blockIdx.y * blockDim.y + threadIdx.y; // height
	const int z = blockIdx.z * blockDim.z + threadIdx.z; // channel

	if (i >= height || j >= width) {
		return;
	}
	// TODO pass instead of hardcoding. Maybe from a file??
	int size = 7; // Only odd numbers
	int mat[9] = { 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9 };

	float accumulator = 0;

	int s = (size - 1) / 2;
	int ac = 0;
	for (int k = -s; k <= s; ++k) {
		for (int l = -s; l <= s; ++l) {
			accumulator += d_img[(width * height * z) + (i * width + k) + j + l] * 1/49;
			ac++;
		}
	}

	d_img_conv[(width * height * z) + i * width + j] = accumulator;
}

int main() {
	CImg<unsigned char> image("C:\\Users\\hugos\\Downloads\\spacex.bmp"), visu(500, 400, 1, 3, 0);

    // Add vectors in parallel.
    cudaError_t cudaStatus = convImage(&image);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

void freeResources(int *dev_a, int *dev_b, int *dev_c) {
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t convImage(CImg<unsigned char> *img) {
	// Number of pixels in a single channel
	unsigned int channelSize = img->width() * img->height();
	// Number of pixels in total (width * height * channels)
	unsigned int tSize = img->size();

	// Original image in GPU
    float *d_img = 0;
	// Convoluted image in GPU
    float *d_img_conv = 0;
	// Original image in CPU
	float *h_img = (float*) malloc(tSize * sizeof(float));
	// Convoluted image in CPU
	float *h_img_conv = (float*) malloc(tSize * sizeof(float));


	// Fill in h_img with the img
	// CImg format is R1R2R3R4...G1G2...B1B1
	for (int i = 0; i < img->height(); ++i) {
		for (int j = 0; j < img->width(); ++j) {
			for (int k = 0; k < img->spectrum(); ++k) {
				h_img[(channelSize * k) + i * img->width() + j] = img->operator()(j, i, k, 0);
			}
		}
	}

	// Display original image as a reference
	CImg<unsigned char> img2(h_img, img->width(), img->height(), 1, img->spectrum(), false);
	CImgDisplay main_disp2(img2);

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		//freeResources(d_img);
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**) &d_img, tSize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
		//freeResources(d_img, d_img_conv);
    }

    cudaStatus = cudaMalloc((void**)&d_img_conv, tSize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
		//freeResources(d_img, d_img_conv);
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(d_img, h_img, tSize * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
		//freeResources(dev_a, dev_b, dev_c);
    }

    // Launch a kernel on the GPU
	const unsigned char bSize = 16;
	dim3 dimBlock(bSize, bSize, img->spectrum());
	dim3 dimGrid((img->width()) / bSize, (img->height()) / bSize);
    convolution<<<dimGrid, dimBlock>>>(d_img, d_img_conv, img->width(), img->height());

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//freeResources(dev_a, dev_b, dev_c);
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		//freeResources(dev_a, dev_b, dev_c);
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(h_img_conv, d_img_conv, tSize * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
		//freeResources(dev_a, dev_b, dev_c);
    }

	// Display result
	CImg<unsigned char> img_conv(h_img_conv, img->width(), img->height(), 1, img->spectrum(), false);
	CImgDisplay main_disp(img_conv);
	while(true){}

    return cudaStatus;
}
