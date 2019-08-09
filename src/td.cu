
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <stdio.h>
#include <math.h>

#define N (1<<20)

int main(int argc, char** argv) {

	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, NULL);

	thrust::device_vector<float> dvec_x(N);
	thrust::device_vector<float> dvec_y(N);

}
