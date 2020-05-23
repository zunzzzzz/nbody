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
namespace param {
    const int n_steps = 200000;
    const double dt = 60;
    const double eps = 1e-3;
    const double G = 6.674e-11;
    double gravity_device_mass(double m0, double t) {
        return m0 + 0.5 * m0 * fabs(sin(t / 6000));
    }
    const double planet_radius = 1e7;
    const double missile_speed = 1e6;
    double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
}  // namespace param
void read_input(const char* filename, int& n, int& planet, int& asteroid,
    double*& qx, double*& qy, double*& qz,
    double*& vx, double*& vy, double*& vz,
    double*& m, int*& type, int& count) 
{
    std::ifstream fin(filename);
    std::string tmp_type;
    fin >> n >> planet >> asteroid;
    if(count == 0) {
        qx = new double[n];
        qy = new double[n];
        qz = new double[n];
        vx = new double[n];
        vy = new double[n];
        vz = new double[n];
        m = new double[n];
        type = new int[n];
    }
    count++;
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

void write_output(const char* filename, double min_dist, int hit_time_step,
    int gravity_device_id, double missile_cost) {
    std::ofstream fout(filename);
    fout << std::scientific
         << std::setprecision(std::numeric_limits<double>::digits10 + 1) << min_dist
         << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}
__global__
void run_step(int step, int n, double* qx, double* qy,
    double* qz, double* vx, double* vy,
    double* vz, double* m,
    int* type, double *ax, double *ay, double *az) 
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        ax[i] = 0;
        ay[i] = 0;
        az[i] = 0;
    }
    __syncthreads();
    // compute accelerations
    for (int i = index; i < n; i += stride) {
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            double mj = m[j];
            // if (type[j] == "device") {
            if (type[j] == 1) {
                // mj = param::gravity_device_mass(mj, step * param::dt);
                mj = mj + 0.5 * mj * fabs(sin((double) 60 / (double)6000));
            }
            double dx = qx[j] - qx[i];
            double dy = qy[j] - qy[i];
            double dz = qz[j] - qz[i];
            // double dist3 = pow(dx * dx + dy * dy + dz * dz + param::eps * param::eps, 1.5);
            double dist3 = pow(dx * dx + dy * dy + dz * dz + 1e-3 * 1e-3, 1.5);
            ax[i] += 6.674e-11 * mj * dx / dist3;
            ay[i] += 6.674e-11 * mj * dy / dist3;
            az[i] += 6.674e-11 * mj * dz / dist3;
        }
    }
    __syncthreads();
    // update velocities
    for (int i = index; i < n; i += stride) {
        vx[i] += ax[i] * 60;
        vy[i] += ay[i] * 60;
        vz[i] += az[i] * 60;
    }
    __syncthreads();
    // update positions
    for (int i = index; i < n; i += stride) {
        qx[i] += vx[i] * 60;
        qy[i] += vy[i] * 60;
        qz[i] += vz[i] * 60;
    }
    __syncthreads();
}
int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }
    int n, planet, asteroid, count = 0;
    double *qx, *qy, *qz, *vx, *vy, *vz, *m;
    double *qx_gpu, *qy_gpu, *qz_gpu, *vx_gpu, *vy_gpu, *vz_gpu, *m_gpu, *ax, *ay, *az;
    int* type; // 1 for device 0 for non-device
    int* type_gpu; // 1 for device 0 for non-device
    clock_t t1, t2;
    size_t threads_per_block = 256;
    size_t num_of_blocks = 32 * 20;

    int device_id;
    cudaGetDevice(&device_id);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    int num_of_SMs = props.multiProcessorCount;
    std::cout << num_of_SMs << std::endl;



    // Problem 1
    t1 = clock();
    double min_dist = std::numeric_limits<double>::infinity();
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type, count);
    cudaMalloc(&qx_gpu, n * sizeof(double));
    cudaMalloc(&qy_gpu, n * sizeof(double));
    cudaMalloc(&qz_gpu, n * sizeof(double));
    cudaMalloc(&vx_gpu, n * sizeof(double));
    cudaMalloc(&vy_gpu, n * sizeof(double));
    cudaMalloc(&vz_gpu, n * sizeof(double));
    cudaMalloc(&m_gpu, n * sizeof(double));
    cudaMalloc(&type_gpu, n * sizeof(int));
    cudaMalloc(&ax, n * sizeof(double));
    cudaMalloc(&ay, n * sizeof(double));
    cudaMalloc(&az, n * sizeof(double));
    for (int i = 0; i < n; i++) {
        if (type[i] == 1) {
            m[i] = 0;
        }
    }
    
    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            cudaMemcpy(qx_gpu, qx, n * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(qy_gpu, qy, n * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(qz_gpu, qz, n * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(vx_gpu, vx, n * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(vy_gpu, vy, n * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(vz_gpu, vz, n * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(m_gpu, m, n * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(type_gpu, type, n * sizeof(int), cudaMemcpyHostToDevice);
            run_step<<<num_of_blocks, threads_per_block>>>(step, n, qx_gpu, qy_gpu, qz_gpu, vx_gpu, vy_gpu, vz_gpu, m_gpu, type_gpu, ax, ay, az);
            cudaMemcpy(qx, qx_gpu, n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(qy, qy_gpu, n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(qz, qz_gpu, n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(vx, vx_gpu, n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(vy, vy_gpu, n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(vz, vz_gpu, n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(m, m_gpu, n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(type, type_gpu, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
        }
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        // std::cout << qx[planet] << " " << qx[asteroid] << std::endl;
        min_dist = std::min(min_dist, sqrt(dx * dx + dy * dy + dz * dz));
    }
    t2 = clock();
    std::cout << "min_dist " << min_dist << std::endl;
    std::cout << "Problem 1 cost " << (double)((t2 - t1) / CLOCKS_PER_SEC) << " seconds" << std::endl;
    // // Problem 2
    // t1 = clock();
    // int hit_time_step = -2;
    // read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type, count);
    // cudaMemcpy(qx_gpu, qx, n * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(qy_gpu, qy, n * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(qz_gpu, qz, n * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(vx_gpu, vx, n * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(vy_gpu, vy, n * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(vz_gpu, vz, n * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(m_gpu, m, n * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(type_gpu, type, n * sizeof(int), cudaMemcpyHostToDevice);
    // for (int step = 0; step <= param::n_steps; step++) {
    //     if (step > 0) {
    //         run_step<<<num_of_blocks, threads_per_block>>>(step, n, qx, qy, qz, vx, vy, vz, m, type);
    //         cudaDeviceSynchronize();
    //     }
    //     double dx = qx[planet] - qx[asteroid];
    //     double dy = qy[planet] - qy[asteroid];
    //     double dz = qz[planet] - qz[asteroid];
    //     if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
    //         hit_time_step = step;
    //         break;
    //     }
    // }
    // cudaDeviceSynchronize();
    // t2 = clock();
    // std::cout << "Problem 2 cost " << (double)((t2 - t1) / CLOCKS_PER_SEC) << " seconds" << std::endl;
    // // Problem 3
    // t1 = clock();
    // std::vector<int> device_index;
    // int gravity_device_id = -999;
    // bool is_saved = false;
    // bool is_explode = false;
    // double missile_cost = std::numeric_limits<double>::infinity();
    // double missel_travel_distance = 0;
    // read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type, count);
    // for(int iter = 0; iter < n; iter++) {
    //     // if (type[iter] == "device") {
    //     if (type[iter] == 1) {
    //         device_index.push_back(iter);
    //     }
    // }
    // for(int iter = 0; iter < device_index.size(); iter++) {
    //     // initialize
    //     read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type, count);
    //     cudaMemcpy(qx_gpu, qx, n * sizeof(double), cudaMemcpyHostToDevice);
    //     cudaMemcpy(qy_gpu, qy, n * sizeof(double), cudaMemcpyHostToDevice);
    //     cudaMemcpy(qz_gpu, qz, n * sizeof(double), cudaMemcpyHostToDevice);
    //     cudaMemcpy(vx_gpu, vx, n * sizeof(double), cudaMemcpyHostToDevice);
    //     cudaMemcpy(vy_gpu, vy, n * sizeof(double), cudaMemcpyHostToDevice);
    //     cudaMemcpy(vz_gpu, vz, n * sizeof(double), cudaMemcpyHostToDevice);
    //     cudaMemcpy(m_gpu, m, n * sizeof(double), cudaMemcpyHostToDevice);
    //     cudaMemcpy(type_gpu, type, n * sizeof(int), cudaMemcpyHostToDevice);
    //     double tmp = m[device_index[iter]];
    //     double time = 0;
    //     missel_travel_distance = 0;
    //     is_explode = false;
    //     int step = 0;
    //     for (step = 0; step <= param::n_steps; step++) {
    //         if (step > 0) {
    //             run_step<<<num_of_blocks, threads_per_block>>>(step, n, qx, qy, qz, vx, vy, vz, m, type);
    //             cudaDeviceSynchronize();
    //         }
    //         double dx = qx[planet] - qx[asteroid];
    //         double dy = qy[planet] - qy[asteroid];
    //         double dz = qz[planet] - qz[asteroid];
    //         if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
    //             break;
    //         }
    //         if(is_explode == false) {
    //             double target_dx = qx[planet] - qx[device_index[iter]];
    //             double target_dy = qy[planet] - qy[device_index[iter]];
    //             double target_dz = qz[planet] - qz[device_index[iter]];
    //             missel_travel_distance += param::missile_speed * param::dt;
    //             time += param::dt;
    //             if (missel_travel_distance * missel_travel_distance > target_dx * target_dx + target_dy * target_dy + target_dz * target_dz) {
    //                 is_explode = true;
    //                 m[device_index[iter]] = 0;
    //             }
    //         }
    //     }
    //     if(step == param::n_steps + 1) {
    //         is_saved = true;
    //         double cost = param::get_missile_cost(time);
    //         if(cost < missile_cost) {
    //             missile_cost = cost;
    //             gravity_device_id = device_index[iter];
    //         }
    //     }
    //     m[device_index[iter]] = tmp;
    // }
    // if(!is_saved) {
    //     gravity_device_id = -1;
    //     missile_cost = 0;
    // }
    // t2 = clock();
    // std::cout << "Problem 3 cost " << (double)((t2 - t1) / CLOCKS_PER_SEC) << " seconds" << std::endl;
    // write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
    // free memory
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
    cudaFree(ax);
    cudaFree(ay);
    cudaFree(az);
    return 0;
}
