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
#include <pthread.h>
__device__ double min_dist_gpu;
__device__ double target_distance_gpu;
__device__ double missel_travel_distance_gpu;
__device__ double time_gpu;
__device__ double missile_cost_gpu;
__device__ bool is_hit_gpu = false;
__device__ bool is_explode_gpu = false;
__device__ bool is_saved_gpu = false;
__device__ int hit_time_step_gpu = -2;
__device__ int hit_time_step_p3_gpu = -2;
__device__ int gravity_device_id_gpu;
namespace param {
    const int n_steps = 200000;
}  // namespace param
class Missile{
public:
    int index;
    double cost;
    char* file_name;
};
void read_input(const char* filename, int& n, int& planet, int& asteroid,
    double*& qx, double*& qy, double*& qz,
    double*& vx, double*& vy, double*& vz,
    double*& m, bool*& is_device) 
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
    is_device = new bool[n];
    for (int i = 0; i < n; i++) {
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> tmp_type;
        if(tmp_type == "device") {
            is_device[i] = true;
        }
        else {
            is_device[i] = false;
        }
    }
}
void free_memory(double*& qx, double*& qy, double*& qz, double*& vx, double*& vy, double*& vz, double*& m, bool*& is_device,
                double*& qx_gpu, double*& qy_gpu, double*& qz_gpu, double*& vx_gpu, double*& vy_gpu, double*& vz_gpu, double*& m_gpu, bool*& is_device_gpu) 
{
    delete(qx);
    delete(qy);
    delete(qz);
    delete(vx);
    delete(vy);
    delete(vz);
    delete(m);
    delete(is_device); 
    cudaFree(qx_gpu);
    cudaFree(qy_gpu);
    cudaFree(qz_gpu);
    cudaFree(vx_gpu);
    cudaFree(vy_gpu);
    cudaFree(vz_gpu);
    cudaFree(m_gpu);
    cudaFree(is_device_gpu);
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
void calculate_missile_cost(int device)
{
    if(hit_time_step_p3_gpu == -2) {
        is_saved_gpu = true;
        double cost = 1e5 + 1e3 * time_gpu;
        if(cost < missile_cost_gpu) {
            missile_cost_gpu = cost;
            gravity_device_id_gpu = device;
        }
    }
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
    }
}
__global__
void calculate_hit_p3(int step, int planet, int asteroid, double* qx, double* qy, double* qz)
{
    double dx = qx[planet] - qx[asteroid];
    double dy = qy[planet] - qy[asteroid];
    double dz = qz[planet] - qz[asteroid];
    if(sqrt(dx * dx + dy * dy + dz * dz) < 1e7 && is_hit_gpu == false) {
        is_hit_gpu = true;
        hit_time_step_p3_gpu = step;
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
void p3_init()
{
    hit_time_step_p3_gpu = -2;
    missel_travel_distance_gpu = 0;
    time_gpu = 0;
    is_explode_gpu = false;
    is_hit_gpu = false;
}
__global__
void set_mass(double* m, bool* is_device, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        if (is_device[i]) m[i] = 0;
    }
}
__global__
void run_step(int step, const int n, double* qx, double* qy,
    double* qz, double* vx, double* vy,
    double* vz, double* m,
    bool* is_device) 
{   
    // blockDim means block size, gridDim means number of blocks
    __shared__ double ax, ay, az;
    if(threadIdx.x == 0) { 
        ax = 0;
        ay = 0;
        az = 0;
    }
    __syncthreads();
    // compute accelerationsn
    for(int i = blockIdx.x; i < n; i += gridDim.x) {
        for(int j = threadIdx.x; j < n; j += blockDim.x) {
            if(i != j) {
                double mj = m[j];
                if (is_device[j]) {
                    mj = mj + 0.5 * mj * fabs(sin((double) (step) / (double)100));
                }
                double dx = qx[j] - qx[i];
                double dy = qy[j] - qy[i];
                double dz = qz[j] - qz[i];
                double dist3 = pow(dx * dx + dy * dy + dz * dz + 1e-6, 1.5);

                atomicAdd(&ax, 6.674e-11 * mj * dx / dist3);
                atomicAdd(&ay, 6.674e-11 * mj * dy / dist3);
                atomicAdd(&az, 6.674e-11 * mj * dz / dist3);
            }
        }
    }
    __syncthreads();
    // update velocities
    if(threadIdx.x == 0) { 
        vx[blockIdx.x] += ax * 60;
        vy[blockIdx.x] += ay * 60;
        vz[blockIdx.x] += az * 60;
    }
    // update positions
    if(threadIdx.x == 0) { 
        qx[blockIdx.x] += vx[blockIdx.x] * 60;    
        qy[blockIdx.x] += vy[blockIdx.x] * 60;
        qz[blockIdx.x] += vz[blockIdx.x] * 60;
    }
}
void* p3(void* missile_data_passed) {
    cudaSetDevice(1);
    int n, planet, asteroid;
    double *qx, *qy, *qz, *vx, *vy, *vz, *m;
    bool* is_device; 
    // gpu pointer
    double *qx_gpu, *qy_gpu, *qz_gpu, *vx_gpu, *vy_gpu, *vz_gpu, *m_gpu;
    bool *is_device_gpu;
    // gpu variable
    size_t threads_per_block, num_of_blocks;

    Missile* missile_data_tmp = (Missile*) missile_data_passed;
    missile_data_tmp -> cost = std::numeric_limits<double>::infinity();
    missile_data_tmp -> index = -999;

    read_input(missile_data_tmp->file_name, n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, is_device);
    threads_per_block = n;
    num_of_blocks = n;
    cudaMalloc(&qx_gpu, n * sizeof(double));
    cudaMalloc(&qy_gpu, n * sizeof(double));
    cudaMalloc(&qz_gpu, n * sizeof(double));
    cudaMalloc(&vx_gpu, n * sizeof(double));
    cudaMalloc(&vy_gpu, n * sizeof(double));
    cudaMalloc(&vz_gpu, n * sizeof(double));
    cudaMalloc(&m_gpu, n * sizeof(double));
    cudaMalloc(&is_device_gpu, n * sizeof(bool));

    std::vector<int> device_index; // use to iterate all device
    bool is_saved = false;
    // record device index
    for(int iter = 0; iter < n; iter++) {
        if (is_device[iter]) {
            device_index.push_back(iter);
        }
    }
    cudaMemcpyToSymbol(missile_cost_gpu, &(missile_data_tmp -> cost), sizeof(double), 0, cudaMemcpyHostToDevice);
    for(int iter = 0; iter < device_index.size(); iter++) {
        // initialize
        cudaMemcpy(qx_gpu, qx, n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(qy_gpu, qy, n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(qz_gpu, qz, n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(vx_gpu, vx, n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(vy_gpu, vy, n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(vz_gpu, vz, n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(m_gpu, m, n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(is_device_gpu, is_device, n * sizeof(bool), cudaMemcpyHostToDevice);
        p3_init<<<1, 1>>>();
        for (int step = 0; step <= 200000; step++) {
            if (step > 0) {
                run_step<<<num_of_blocks, threads_per_block, 0>>>(step, n, qx_gpu, qy_gpu, qz_gpu, vx_gpu, vy_gpu, vz_gpu, m_gpu, is_device_gpu);
                cudaError_t err;
                err = cudaGetLastError(); // `cudaGetLastError` will return the error from above.
                if (err != cudaSuccess)
                {
                    printf("Error: %s\n", cudaGetErrorString(err));
                }
            }
            calculate_hit_p3<<<1, 1, 0>>>(step, planet, asteroid, qx_gpu, qy_gpu, qz_gpu);
            calculate_target_distance<<<1, 1, 0>>>(planet, device_index[iter], qx_gpu, qy_gpu, qz_gpu, m_gpu);
        }
        calculate_missile_cost<<<1, 1, 0>>>(device_index[iter]);
    }
    cudaMemcpyFromSymbol(&(missile_data_tmp -> cost), missile_cost_gpu, sizeof(double), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&(missile_data_tmp -> index), gravity_device_id_gpu, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&is_saved, is_saved_gpu, sizeof(bool), 0, cudaMemcpyDeviceToHost);
    if(!is_saved) {
        (missile_data_tmp -> index) = -1;
        (missile_data_tmp -> cost) = 0;
    }
    pthread_exit(NULL);
}
int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }
    int n, planet, asteroid;
    double *qx, *qy, *qz, *vx, *vy, *vz, *m;
    bool* is_device; 
    // gpu pointer
    double *qx_gpu, *qy_gpu, *qz_gpu, *vx_gpu, *vy_gpu, *vz_gpu, *m_gpu;
    bool *is_device_gpu; 
    // gpu variable
    size_t threads_per_block, num_of_blocks;

    // problem 3 using pthread
    Missile missile_data;
    missile_data.file_name = argv[1];
    pthread_t t; 
    pthread_create(&t, NULL, p3, (void*) &missile_data); 

    // Problem 1
    cudaSetDevice(0);
    double min_dist = std::numeric_limits<double>::infinity();
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, is_device);
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
    cudaMalloc(&is_device_gpu, n * sizeof(bool));
    // copy memory to gpu
    cudaMemcpy(qx_gpu, qx, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(qy_gpu, qy, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(qz_gpu, qz, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vx_gpu, vx, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vy_gpu, vy, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m_gpu, m, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vz_gpu, vz, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(is_device_gpu, is_device, n * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(min_dist_gpu, &min_dist, sizeof(double), 0, cudaMemcpyHostToDevice);
    // set device mass to zero
    set_mass<<<num_of_blocks, threads_per_block>>>(m_gpu, is_device_gpu, n);
    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            run_step<<<num_of_blocks, threads_per_block, 0>>>(step, n, qx_gpu, qy_gpu, qz_gpu, vx_gpu, vy_gpu, vz_gpu, m_gpu, is_device_gpu);
        }
        calculate_min<<<1, 1, 0>>>(planet, asteroid, qx_gpu, qy_gpu, qz_gpu);
    }
    cudaMemcpyFromSymbol(&min_dist, min_dist_gpu, sizeof(double), 0, cudaMemcpyDeviceToHost);
    std::cout << min_dist << std::endl;
    // Problem 2
    int hit_time_step = -2;
    // copy memory to gpu
    cudaMemcpy(qx_gpu, qx, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(qy_gpu, qy, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(qz_gpu, qz, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vx_gpu, vx, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vy_gpu, vy, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vz_gpu, vz, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m_gpu, m, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(is_device_gpu, is_device, n * sizeof(bool), cudaMemcpyHostToDevice);
    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            run_step<<<num_of_blocks, threads_per_block, 0>>>(step, n, qx_gpu, qy_gpu, qz_gpu, vx_gpu, vy_gpu, vz_gpu, m_gpu, is_device_gpu);
        }
        calculate_hit<<<1, 1, 0>>>(step, planet, asteroid, qx_gpu, qy_gpu, qz_gpu);
    }
    cudaMemcpyFromSymbol(&hit_time_step, hit_time_step_gpu, sizeof(int), 0, cudaMemcpyDeviceToHost);
    // if planet is safe (hit_time_step == -2)
    pthread_join(t, NULL); 
    if(hit_time_step == -2) {
        // gravity_device_id = -1;
        // missile_cost = 0;
        missile_data.index = -1;
        missile_data.cost = 0;
    }
    // write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
    write_output(argv[2], min_dist, hit_time_step, missile_data.index, missile_data.cost);
    // free memory
    free_memory(qx, qy, qz, vx, vy, vz, m, is_device, qx_gpu, qy_gpu, qz_gpu, vx_gpu, vy_gpu, vz_gpu, m_gpu, is_device_gpu);
    return 0;
}
