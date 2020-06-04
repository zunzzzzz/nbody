#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <string>
#include <stdlib.h>
#include <vector>
#include <time.h>
__device__ double min_dist_gpu;
__device__ double target_distance_gpu;
__device__ double missel_travel_distance_gpu;
__device__ double time_gpu;
__device__ bool is_hit_gpu = false;
__device__ bool is_explode_gpu = false;
__device__ int hit_time_step_gpu = -2;
namespace param {
    const int n_steps = 200000;
    // const double dt = 60;
    // const double eps = 1e-3;
    // const double G = 6.674e-11;
    double gravity_device_mass(double m0, double t) {
        return m0 + 0.5 * m0 * fabs(sin(t / 6000));
    }
    // const double planet_radius = 1e7;
    // const double missile_speed = 1e6;
    double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
}  // namespace param
void read_input(const char* filename, int& n, int& planet, int& asteroid,
    double*& qx, double*& qy, double*& qz,
    double*& vx, double*& vy, double*& vz,
    double*& m, int*& type) 
{
    std::ifstream fin(filename);
    std::string tmp_type;
    fin >> n >> planet >> asteroid;
    qx = new double[n];
    qy = new double[n];
    qz = new double[n];
    vx = new double[n];
    vy = new double[n];
    vz = new double[n];
    m = new double[n];
    type = new int[n];
    for (int i = 0; i < n; i++) {
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> tmp_type;
        if(tmp_type == "device") {
            type[i] = 1;
        }
        else {
            type[i] = 0;
        }
    }
}
void free_memory(double*& qx, double*& qy, double*& qz, double*& vx, double*& vy, double*& vz, double*& m, int*& type,
                double*& qx_gpu, double*& qy_gpu, double*& qz_gpu, double*& vx_gpu, double*& vy_gpu, double*& vz_gpu, double*& m_gpu, int*& type_gpu) 
{
    delete(qx);
    delete(qy);
    delete(qz);
    delete(vx);
    delete(vy);
    delete(vz);
    delete(m);
    delete(type); 
    cudaFree(qx_gpu);
    cudaFree(qy_gpu);
    cudaFree(qz_gpu);
    cudaFree(vx_gpu);
    cudaFree(vy_gpu);
    cudaFree(vz_gpu);
    cudaFree(m_gpu);
    cudaFree(type_gpu);
}
void write_output(const char* filename, double min_dist, int hit_time_step,
    int gravity_device_id, double missile_cost) 
{
    std::ofstream fout(filename);
    fout << std::scientific
         << std::setprecision(std::numeric_limits<double>::digits10 + 1) << min_dist
         << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}
// define atomic add in device
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif
__global__
void calculate_min(int planet, int asteroid, double* qx, double* qy, double* qz)
{
    double dx = qx[planet] - qx[asteroid];
    double dy = qy[planet] - qy[asteroid];
    double dz = qz[planet] - qz[asteroid];
    min_dist_gpu = min(min_dist_gpu, sqrt(dx * dx + dy * dy + dz * dz));
}
__global__
void calculate_missle_distance(int planet, int device, double* qx, double* qy, double* qz)
{
    double dx = qx[planet] - qx[device];
    double dy = qy[planet] - qy[device];
    double dz = qz[planet] - qz[device];
    target_distance_gpu = sqrt(dx * dx + dy * dy + dz * dz);
}
__global__
void calculate_hit(int step, int planet, int asteroid, double* qx, double* qy, double* qz)
{
    double dx = qx[planet] - qx[asteroid];
    double dy = qy[planet] - qy[asteroid];
    double dz = qz[planet] - qz[asteroid];
    if(sqrt(dx * dx + dy * dy + dz * dz) < 1e7 && is_hit_gpu == false) {
        is_hit_gpu = true;
        hit_time_step_gpu = step;
        printf("hit\n");
    }
}
__global__
void calculate_target_distance(int planet, int device, double* qx, double* qy, double* qz, double* m)
{
    if(is_explode_gpu == false) {
        missel_travel_distance_gpu += 1e6 * 60;
        time_gpu += 60;
        double dx = qx[planet] - qx[device];
        double dy = qy[planet] - qy[device];
        double dz = qz[planet] - qz[device];
        target_distance_gpu = sqrt(dx * dx + dy * dy + dz * dz);
        if (missel_travel_distance_gpu > target_distance_gpu) {
            is_explode_gpu = true;
            m[device] = 0;
        }
    }
}
__global__
void set_mass(double* m, int* type, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        if (type[i] == 1) {
            m[i] = 0;
        }
    }
}
__global__
void run_step(int step, const int n, double* qx, double* qy,
    double* qz, double* vx, double* vy,
    double* vz, double* m,
    int* type) 
{   
    // blockDim means block size, gridDim means number of blocks
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    __shared__ double ax[40];
    __shared__ double ay[40];
    __shared__ double az[40];
    for(int i = blockIdx.x; i < n; i += gridDim.x) {
        if(threadIdx.x == 0) { 
            ax[i] = 0;
            ay[i] = 0;
            az[i] = 0;
        }
    }
    __syncthreads();
    
    // compute accelerationsn
    for(int i = blockIdx.x; i < n; i += gridDim.x) {
        for(int j = threadIdx.x; j < n; j += blockDim.x) {
            if(i != j) {
                double mj = m[j];
                if (type[j] == 1) {
                    mj = mj + 0.5 * mj * fabs(sin((double) (step * 60) / (double)6000));
                }
                double dx = qx[j] - qx[i];
                double dy = qy[j] - qy[i];
                double dz = qz[j] - qz[i];
                double dist3 = pow(dx * dx + dy * dy + dz * dz + 1e-6, 1.5);

                atomicAdd(&ax[i], 6.674e-11 * mj * dx / dist3);
                atomicAdd(&ay[i], 6.674e-11 * mj * dy / dist3);
                atomicAdd(&az[i], 6.674e-11 * mj * dz / dist3);
            }
        }
    }
    __syncthreads();
    // update velocities
    for(int i = blockIdx.x; i < n; i += gridDim.x) {
        if(threadIdx.x == 0) { 
            vx[i] += ax[i] * 60;
            vy[i] += ay[i] * 60;
            vz[i] += az[i] * 60;
        }
    }
    __syncthreads();
    // update positions
    for(int i = blockIdx.x; i < n; i += gridDim.x) {
        if(threadIdx.x == 0) { 
            qx[i] += vx[i] * 60;    
            qy[i] += vy[i] * 60;
            qz[i] += vz[i] * 60;
        }
    }
}
int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }
    int n, planet, asteroid;
    double *qx, *qy, *qz, *vx, *vy, *vz, *m;
    double *qx_gpu, *qy_gpu, *qz_gpu, *vx_gpu, *vy_gpu, *vz_gpu, *m_gpu;
    int* type; // 1 for device 0 for non-device
    int* type_gpu; // 1 for device 0 for non-device
    clock_t t1, t2;


    int device_id;
    cudaGetDevice(&device_id);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    int num_of_SMs = props.multiProcessorCount;
    std::cout << num_of_SMs << std::endl;


   
    // Problem 1
    t1 = clock();
    double min_dist = std::numeric_limits<double>::infinity();
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    size_t threads_per_block, num_of_blocks;
    threads_per_block = n;
    num_of_blocks = n;
    // allocate memory
    cudaMalloc(&qx_gpu, n * sizeof(double));
    cudaMalloc(&qy_gpu, n * sizeof(double));
    cudaMalloc(&qz_gpu, n * sizeof(double));
    cudaMalloc(&vx_gpu, n * sizeof(double));
    cudaMalloc(&vy_gpu, n * sizeof(double));
    cudaMalloc(&vz_gpu, n * sizeof(double));
    cudaMalloc(&m_gpu, n * sizeof(double));
    cudaMalloc(&type_gpu, n * sizeof(int));
    // copy memory to gpu
    cudaMemcpy(qx_gpu, qx, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(qy_gpu, qy, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(qz_gpu, qz, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vx_gpu, vx, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vy_gpu, vy, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vz_gpu, vz, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m_gpu, m, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(type_gpu, type, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(min_dist_gpu, &min_dist, sizeof(double), 0, cudaMemcpyHostToDevice);
    // set device mass to zero
    set_mass<<<num_of_blocks, threads_per_block>>>(m_gpu, type_gpu, n);
    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            run_step<<<num_of_blocks, threads_per_block>>>(step, n, qx_gpu, qy_gpu, qz_gpu, vx_gpu, vy_gpu, vz_gpu, m_gpu, type_gpu);
            // cudaError_t err;
            // err = cudaGetLastError(); // `cudaGetLastError` will return the error from above.
            // if (err != cudaSuccess)
            // {
            //     printf("Error: %s\n", cudaGetErrorString(err));
            // }
        }
        calculate_min<<<1, 1>>>(planet, asteroid, qx_gpu, qy_gpu, qz_gpu);
    }
    cudaMemcpyFromSymbol(&min_dist, min_dist_gpu, sizeof(double), 0, cudaMemcpyDeviceToHost);
    t2 = clock();
    std::cout << "min_dist " << min_dist << std::endl;
    std::cout << "Problem 1 cost " << (double)((t2 - t1) / CLOCKS_PER_SEC) << " seconds" << std::endl;
    // Problem 2
    t1 = clock();
    int hit_time_step = -2;
    bool is_hit = false;
    cudaMemcpy(qx_gpu, qx, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(qy_gpu, qy, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(qz_gpu, qz, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vx_gpu, vx, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vy_gpu, vy, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vz_gpu, vz, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m_gpu, m, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(type_gpu, type, n * sizeof(int), cudaMemcpyHostToDevice);
    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            run_step<<<num_of_blocks, threads_per_block>>>(step, n, qx_gpu, qy_gpu, qz_gpu, vx_gpu, vy_gpu, vz_gpu, m_gpu, type_gpu);
        }
        calculate_hit<<<1, 1>>>(step, planet, asteroid, qx_gpu, qy_gpu, qz_gpu);
    }
    cudaMemcpyFromSymbol(&hit_time_step, hit_time_step_gpu, sizeof(int), 0, cudaMemcpyDeviceToHost);
    t2 = clock();
    std::cout << "hit_time_step " << hit_time_step << std::endl;
    std::cout << "Problem 2 cost " << (double)((t2 - t1) / CLOCKS_PER_SEC) << " seconds" << std::endl;
    // Problem 3
    t1 = clock();
    std::vector<int> device_index;
    int gravity_device_id = -999;
    bool is_saved = false;
    bool is_explode = false;
    double missile_cost = std::numeric_limits<double>::infinity();
    double missel_travel_distance = 0;
    if(hit_time_step == -2) {
        gravity_device_id = -1;
        missile_cost = 0;
    }
    else {
        for(int iter = 0; iter < n; iter++) {
            if (type[iter] == 1) {
                device_index.push_back(iter);
            }
        }
        for(int iter = 0; iter < device_index.size(); iter++) {
            // initialize
            double tmp = m[device_index[iter]];
            double time = 0;
            int hit_time_step_p3 = -2;
            is_hit = false; // initialize to false
            missel_travel_distance = 0;
            is_explode = false;
            cudaMemcpy(qx_gpu, qx, n * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(qy_gpu, qy, n * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(qz_gpu, qz, n * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(vx_gpu, vx, n * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(vy_gpu, vy, n * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(vz_gpu, vz, n * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(m_gpu, m, n * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(type_gpu, type, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(hit_time_step_gpu, &hit_time_step_p3, sizeof(int), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(missel_travel_distance_gpu, &missel_travel_distance, sizeof(double), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(time_gpu, &time, sizeof(double), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(is_explode_gpu, &is_explode, sizeof(bool), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(is_hit_gpu, &is_hit, sizeof(bool), 0, cudaMemcpyHostToDevice);
            for (int step = 0; step <= param::n_steps; step++) {
                if (step > 0) {
                    run_step<<<num_of_blocks, threads_per_block>>>(step, n, qx_gpu, qy_gpu, qz_gpu, vx_gpu, vy_gpu, vz_gpu, m_gpu, type_gpu);
                    
                }
                calculate_hit<<<1, 1>>>(step, planet, asteroid, qx_gpu, qy_gpu, qz_gpu);
                calculate_target_distance<<<1, 1>>>(planet, device_index[iter], qx_gpu, qy_gpu, qz_gpu, m_gpu);
            }
            cudaMemcpyFromSymbol(&hit_time_step_p3, hit_time_step_gpu, sizeof(int), 0, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(&time, time_gpu, sizeof(double), 0, cudaMemcpyDeviceToHost);
            if(hit_time_step_p3 == -2) {
                is_saved = true;
                printf("SAFE\n");
                double cost = param::get_missile_cost(time);
                if(cost < missile_cost) {
                    missile_cost = cost;
                    gravity_device_id = device_index[iter];
                }
            }
        }
        if(!is_saved) {
            gravity_device_id = -1;
            missile_cost = 0;
        }
    }
    t2 = clock();
    std::cout << "gravity_device_id " << gravity_device_id << std::endl;
    std::cout << "missile_cost " << missile_cost << std::endl;
    std::cout << "Problem 3 cost " << (double)((t2 - t1) / CLOCKS_PER_SEC) << " seconds" << std::endl;
    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
    // free memory
    free_memory(qx, qy, qz, vx, vy, vz, m, type, qx_gpu, qy_gpu, qz_gpu, vx_gpu, vy_gpu, vz_gpu, m_gpu, type_gpu);
    return 0;
}
