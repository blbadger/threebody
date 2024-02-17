#include <stdio.h>
#include <iostream>
#include <chrono>

// kernal declaration
__global__
void divergence(int n, 
              int steps,
              double delta_t,
              int *times,
              double m_1, double m_2, double m_3,
              double critical_distance,
              double *p1_x, double *p1_y, double *p1_z, 
              double *p2_x, double *p2_y, double *p2_z, 
              double *p3_x, double *p3_y, double *p3_z, 
              double *p1_prime_x, double *p1_prime_y, double *p1_prime_z, 
              double *p2_prime_x, double *p2_prime_y, double *p2_prime_z, 
              double *p3_prime_x, double *p3_prime_y, double *p3_prime_z, 
              double *v1_x, double *v1_y, double *v1_z,
              double *v2_x, double *v2_y, double *v2_z,
              double *v3_x, double *v3_y, double *v3_z,
              double *v1_prime_x, double *v1_prime_y, double *v1_prime_z,
              double *v2_prime_x, double *v2_prime_y, double *v2_prime_z,
              double *v3_prime_x, double *v3_prime_y, double *v3_prime_z)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int still_together = 1;
  int not_diverged = 1;
  int times_ind = times[i];
  double p1x = p1_x[i];
  double p1y = p1_y[i];
  double p1z = p1_z[i];
  double p1_primex = p1_prime_x[i];
  double p1_primey = p1_prime_y[i];
  double p1_primez = p1_prime_z[i];

  double v1x = v1_x[i];
  double v1y = v1_y[i];
  double v1z = v1_z[i];
  double v1_primex = v1_prime_x[i];
  double v1_primey = v1_prime_y[i];
  double v1_primez = v1_prime_z[i];

  double p2x = p2_x[i];
  double p2y = p2_y[i];
  double p2z = p2_z[i];
  double p2_primex = p2_prime_x[i];
  double p2_primey = p2_prime_y[i];
  double p2_primez = p2_prime_z[i];

  double v2x = v2_x[i];
  double v2y = v2_y[i];
  double v2z = v2_z[i];
  double v2_primex = v2_prime_x[i];
  double v2_primey = v2_prime_y[i];
  double v2_primez = v2_prime_z[i];

  double p3x = p3_x[i];
  double p3y = p3_y[i];
  double p3z = p3_z[i];
  double p3_primex = p3_prime_x[i];
  double p3_primey = p3_prime_y[i];
  double p3_primez = p3_prime_z[i];

  double v3x = v3_x[i];
  double v3y = v3_y[i];
  double v3z = v3_z[i];
  double v3_primex = v3_prime_x[i];
  double v3_primey = v3_prime_y[i];
  double v3_primez = v3_prime_z[i];

  double dv_1_x, dv_1_y, dv_1_z;
  double dv_2_x, dv_2_y, dv_2_z;
  double dv_3_x, dv_3_y, dv_3_z;

  double nv1x, nv1y, nv1z;
  double nv2x, nv2y, nv2z;
  double nv3x, nv3y, nv3z;

  double nv1_primex, nv1_primey, nv1_primez;
  double nv2_primex, nv2_primey, nv2_primez;
  double nv3_primex, nv3_primey, nv3_primez;

  double dv_1pr_x, dv_1pr_y, dv_1pr_z;
  double dv_2pr_x, dv_2pr_y, dv_2pr_z;
  double dv_3pr_x, dv_3pr_y, dv_3pr_z;

  if (i < n){
    for (int j=0; j < steps; j++) {
        // compute accelerations
        dv_1_x = -9.8 * m_2 * (p1x - p2x) / pow(sqrt(pow(p1x - p2x, 2) + pow(p1y - p2y, 2) + pow(p1z - p2z, 2)), 3) \
                 -9.8 * m_3 * (p1x - p3x) / pow(sqrt(pow(p1x - p3x, 2) + pow(p1y - p3y, 2) + pow(p1z - p3z, 2)), 3);

        dv_1_y = -9.8 * m_2 * (p1y - p2y) / pow(sqrt(pow(p1x - p2x, 2) + pow(p1y - p2y, 2) + pow(p1z - p2z, 2)), 3) \
                 -9.8 * m_3 * (p1y - p3y) / pow(sqrt(pow(p1x - p3x, 2) + pow(p1y - p3y, 2) + pow(p1z - p3z, 2)), 3);

        dv_1_z = -9.8 * m_2 * (p1z - p2z) / pow(sqrt(pow(p1x - p2x, 2) + pow(p1y - p2y, 2) + pow(p1z - p2z, 2)), 3) \
                 -9.8 * m_3 * (p1z - p3z) / pow(sqrt(pow(p1x - p3x, 2) + pow(p1y - p3y, 2) + pow(p1z - p3z, 2)), 3);

        dv_2_x = -9.8 * m_3 * (p2x - p3x) / pow(sqrt(pow(p2x - p3x, 2) + pow(p2y - p3y, 2) + pow(p2z - p3z, 2)), 3) \
                 -9.8 * m_1 * (p2x - p1x) / pow(sqrt(pow(p2x - p1x, 2) + pow(p2y - p1y, 2) + pow(p2z - p1z, 2)), 3);

        dv_2_y = -9.8 * m_3 * (p2y - p3y) / pow(sqrt(pow(p2x - p3x, 2) + pow(p2y - p3y, 2) + pow(p2z - p3z, 2)), 3) \
                 -9.8 * m_1 * (p2y - p1y) / pow(sqrt(pow(p2x - p1x, 2) + pow(p2y - p1y, 2) + pow(p2z - p1z, 2)), 3);

        dv_2_z = -9.8 * m_3 * (p2z - p3z) / pow(sqrt(pow(p2x - p3x, 2) + pow(p2y - p3y, 2) + pow(p2z - p3z, 2)), 3) \
                 -9.8 * m_1 * (p2z - p1z) / pow(sqrt(pow(p2x - p1x, 2) + pow(p2y - p1y, 2) + pow(p2z - p1z, 2)), 3);

        dv_3_x = -9.8 * m_1 * (p3x - p1x) / pow(sqrt(pow(p3x - p1x, 2) + pow(p3y - p1y, 2) + pow(p3z - p1z, 2)), 3) \
                 -9.8 * m_2 * (p3x - p2x) / pow(sqrt(pow(p3x - p2x, 2) + pow(p3y - p2y, 2) + pow(p3z - p2z, 2)), 3);

        dv_3_y = -9.8 * m_1 * (p3y - p1y) / pow(sqrt(pow(p3x - p1x, 2) + pow(p3y - p1y, 2) + pow(p3z - p1z, 2)), 3) \
                 -9.8 * m_2 * (p3y - p2y) / pow(sqrt(pow(p3x - p2x, 2) + pow(p3y - p2y, 2) + pow(p3z - p2z, 2)), 3);

        dv_3_z = -9.8 * m_1 * (p3z - p1z) / pow(sqrt(pow(p3x - p1x, 2) + pow(p3y - p1y, 2) + pow(p3z - p1z, 2)), 3) \
                 -9.8 * m_2 * (p3z - p2z) / pow(sqrt(pow(p3x - p2x, 2) + pow(p3y - p2y, 2) + pow(p3z - p2z, 2)), 3);


        dv_1pr_x = -9.8 * m_2 * (p1_primex - p2_primex) / pow(sqrt(pow(p1_primex - p2_primex, 2) + pow(p1_primey - p2_primey, 2) + pow(p1_primez - p2_primez, 2)), 3) \
                   -9.8 * m_3 * (p1_primex - p3_primex) / pow(sqrt(pow(p1_primex - p3_primex, 2) + pow(p1_primey - p3_primey, 2) + pow(p1_primez - p3_primez, 2)), 3);

        dv_1pr_y = -9.8 * m_2 * (p1_primey - p2_primey) / pow(sqrt(pow(p1_primex - p2_primex, 2) + pow(p1_primey - p2_primey, 2) + pow(p1_primez - p2_primez, 2)), 3) \
                   -9.8 * m_3 * (p1_primey - p3_primey) / pow(sqrt(pow(p1_primex - p3_primex, 2) + pow(p1_primey - p3_primey, 2) + pow(p1_primez - p3_primez, 2)), 3);

        dv_1pr_z = -9.8 * m_2 * (p1_primez - p2_primez) / pow(sqrt(pow(p1_primex - p2_primex, 2) + pow(p1_primey - p2_primey, 2) + pow(p1_primez - p2_primez, 2)), 3) \
                   -9.8 * m_3 * (p1_primez - p3_primez) / pow(sqrt(pow(p1_primex - p3_primex, 2) + pow(p1_primey - p3_primey, 2) + pow(p1_primez - p3_primez, 2)), 3);

        dv_2pr_x = -9.8 * m_3 * (p2_primex - p3_primex) / pow(sqrt(pow(p2_primex - p3_primex, 2) + pow(p2_primey - p3_primey, 2) + pow(p2_primez - p3_primez, 2)), 3) \
                   -9.8 * m_1 * (p2_primex - p1_primex) / pow(sqrt(pow(p2_primex - p1_primex, 2) + pow(p2_primey - p1_primey, 2) + pow(p2_primez - p1_primez, 2)), 3);

        dv_2pr_y = -9.8 * m_3 * (p2_primey - p3_primey) / pow(sqrt(pow(p2_primex - p3_primex, 2) + pow(p2_primey - p3_primey, 2) + pow(p2_primez - p3_primez, 2)), 3) \
                   -9.8 * m_1 * (p2_primey - p1_primey) / pow(sqrt(pow(p2_primex - p1_primex, 2) + pow(p2_primey - p1_primey, 2) + pow(p2_primez - p1_primez, 2)), 3);

        dv_2pr_z = -9.8 * m_3 * (p2_primez - p3_primez) / pow(sqrt(pow(p2_primex - p3_primex, 2) + pow(p2_primey - p3_primey, 2) + pow(p2_primez - p3_primez, 2)), 3) \
                   -9.8 * m_1 * (p2_primez - p1_primez) / pow(sqrt(pow(p2_primex - p1_primex, 2) + pow(p2_primey - p1_primey, 2) + pow(p2_primez - p1_primez, 2)), 3);

        dv_3pr_x = -9.8 * m_1 * (p3_primex - p1_primex) / pow(sqrt(pow(p3_primex - p1_primex, 2) + pow(p3_primey - p1_primey, 2) + pow(p3_primez - p1_primez, 2)), 3) \
                   -9.8 * m_2 * (p3_primex - p2_primex) / pow(sqrt(pow(p3_primex - p2_primex, 2) + pow(p3_primey - p2_primey, 2) + pow(p3_primez - p2_primez, 2)), 3);

        dv_3pr_y = -9.8 * m_1 * (p3_primey - p1_primey) / pow(sqrt(pow(p3_primex - p1_primex, 2) + pow(p3_primey - p1_primey, 2) + pow(p3_primez - p1_primez, 2)), 3) \
                   -9.8 * m_2 * (p3_primey - p2_primey) / pow(sqrt(pow(p3_primex - p2_primex, 2) + pow(p3_primey - p2_primey, 2) + pow(p3_primez - p2_primez, 2)), 3);

        dv_3pr_z = -9.8 * m_1 * (p3_primez - p1_primez) / pow(sqrt(pow(p3_primex - p1_primex, 2) + pow(p3_primey - p1_primey, 2) + pow(p3_primez - p1_primez, 2)), 3) \
                   -9.8 * m_2 * (p3_primez - p2_primez) / pow(sqrt(pow(p3_primex - p2_primex, 2) + pow(p3_primey - p2_primey, 2) + pow(p3_primez - p2_primez, 2)), 3);

        // find which trajectories have diverged and increment times_ind
        not_diverged = sqrt(pow(p1x - p1_primex, 2) + pow(p1y - p1_primey, 2) + pow(p1z - p1_primez, 2)) <= critical_distance;
        still_together &= not_diverged;
        times_ind = times_ind + still_together;

        // compute new velocities
        nv1x = v1x + delta_t * dv_1_x;
        nv1y = v1y + delta_t * dv_1_y;
        nv1z = v1z + delta_t * dv_1_z;

        nv2x = v2x + delta_t * dv_2_x;
        nv2y = v2y + delta_t * dv_2_y;
        nv2z = v2z + delta_t * dv_2_z;

        nv3x = v3x + delta_t * dv_3_x;
        nv3y = v3y + delta_t * dv_3_y;
        nv3z = v3z + delta_t * dv_3_z;

        nv1_primex = v1_primex + delta_t * dv_1pr_x;
        nv1_primey = v1_primey + delta_t * dv_1pr_y;
        nv1_primez = v1_primez + delta_t * dv_1pr_z;

        nv2_primex = v2_primex + delta_t * dv_2pr_x;
        nv2_primey = v2_primey + delta_t * dv_2pr_y;
        nv2_primez = v2_primez + delta_t * dv_2pr_z;

        nv3_primex = v3_primex + delta_t * dv_3pr_x;
        nv3_primey = v3_primey + delta_t * dv_3pr_y;
        nv3_primez = v3_primez + delta_t * dv_3pr_z;

        // compute positions with current velocities
        p1x = p1x + delta_t * v1x;
        p1y = p1y + delta_t * v1y;
        p1z = p1z + delta_t * v1z;

        p2x = p2x + delta_t * v2x;
        p2y = p2y + delta_t * v2y;
        p2z = p2z + delta_t * v2z;

        p3x = p3x + delta_t * v3x;
        p3y = p3y + delta_t * v3y;
        p3z = p3z + delta_t * v3z;

        p1_primex = p1_primex + delta_t * v1_primex;
        p1_primey = p1_primey + delta_t * v1_primey;
        p1_primez = p1_primez + delta_t * v1_primez;

        p2_primex = p2_primex + delta_t * v2_primex;
        p2_primey = p2_primey + delta_t * v2_primey;
        p2_primez = p2_primez + delta_t * v2_primez;

        p3_primex = p3_primex + delta_t * v3_primex;
        p3_primey = p3_primey + delta_t * v3_primey;
        p3_primez = p3_primez + delta_t * v3_primez;

        // assign new velocities to current velocities
        v1x = nv1x;
        v1y = nv1y;
        v1z = nv1z;

        v2x = nv2x;
        v2y = nv2y;
        v2z = nv2z;

        v3x = nv3x;
        v3y = nv3y;
        v3z = nv3z;

        v1_primex = nv1_primex;
        v1_primey = nv1_primey;
        v1_primez = nv1_primez;

        v2_primex = nv2_primex;
        v2_primey = nv2_primey;
        v2_primez = nv2_primez;

        v3_primex = nv3_primex;
        v3_primey = nv3_primey;
        v3_primez = nv3_primez;
        }

    times[i] = times_ind;
    }
  }

  
int main(void)
{
  int N = 90000;
  int steps = 5000;
  double delta_t = 0.001;
  double critical_distance = 0.5;
  double m1 = 10;
  double m2 = 20;
  double m3 = 30;
  double *p1_x, *p1_y, *p1_z;
  double *p2_x, *p2_y, *p2_z;
  double *p3_x, *p3_y, *p3_z;
  double *p1_prime_x, *p1_prime_y, *p1_prime_z;
  double *p2_prime_x, *p2_prime_y, *p2_prime_z;
  double *p3_prime_x, *p3_prime_y, *p3_prime_z;
  double *v1_x, *v1_y, *v1_z;
  double *v2_x, *v2_y, *v2_z;
  double *v3_x, *v3_y, *v3_z;
  double *v1_prime_x, *v1_prime_y, *v1_prime_z;
  double *v2_prime_x, *v2_prime_y, *v2_prime_z;
  double *v3_prime_x, *v3_prime_y, *v3_prime_z;


  double *d_p1_x, *d_p1_y, *d_p1_z;
  double *d_p2_x, *d_p2_y, *d_p2_z;
  double *d_p3_x, *d_p3_y, *d_p3_z;
  double *d_p1_prime_x, *d_p1_prime_y, *d_p1_prime_z;
  double *d_p2_prime_x, *d_p2_prime_y, *d_p2_prime_z;
  double *d_p3_prime_x, *d_p3_prime_y, *d_p3_prime_z;
  double *d_v1_x, *d_v1_y, *d_v1_z;
  double *d_v2_x, *d_v2_y, *d_v2_z;
  double *d_v3_x, *d_v3_y, *d_v3_z;
  double *d_v1_prime_x, *d_v1_prime_y, *d_v1_prime_z;
  double *d_v2_prime_x, *d_v2_prime_y, *d_v2_prime_z;
  double *d_v3_prime_x, *d_v3_prime_y, *d_v3_prime_z;
  int *times, *d_times;

  p1_x = (double*)malloc(N*sizeof(double));
  p1_y = (double*)malloc(N*sizeof(double));
  p1_z = (double*)malloc(N*sizeof(double));

  p2_x = (double*)malloc(N*sizeof(double));
  p2_y = (double*)malloc(N*sizeof(double));
  p2_z = (double*)malloc(N*sizeof(double));

  p3_x = (double*)malloc(N*sizeof(double));
  p3_y = (double*)malloc(N*sizeof(double));
  p3_z = (double*)malloc(N*sizeof(double));

  p1_prime_x = (double*)malloc(N*sizeof(double));
  p1_prime_y = (double*)malloc(N*sizeof(double));
  p1_prime_z = (double*)malloc(N*sizeof(double));

  p2_prime_x = (double*)malloc(N*sizeof(double));
  p2_prime_y = (double*)malloc(N*sizeof(double));
  p2_prime_z = (double*)malloc(N*sizeof(double));

  p3_prime_x = (double*)malloc(N*sizeof(double));
  p3_prime_y = (double*)malloc(N*sizeof(double));
  p3_prime_z = (double*)malloc(N*sizeof(double));

  v1_x = (double*)malloc(N*sizeof(double));
  v1_y = (double*)malloc(N*sizeof(double));
  v1_z = (double*)malloc(N*sizeof(double));

  v2_x = (double*)malloc(N*sizeof(double));
  v2_y = (double*)malloc(N*sizeof(double));
  v2_z = (double*)malloc(N*sizeof(double));

  v3_x = (double*)malloc(N*sizeof(double));
  v3_y = (double*)malloc(N*sizeof(double));
  v3_z = (double*)malloc(N*sizeof(double));

  v1_prime_x = (double*)malloc(N*sizeof(double));  
  v1_prime_y = (double*)malloc(N*sizeof(double));
  v1_prime_z = (double*)malloc(N*sizeof(double));

  v2_prime_x = (double*)malloc(N*sizeof(double));  
  v2_prime_y = (double*)malloc(N*sizeof(double));
  v2_prime_z = (double*)malloc(N*sizeof(double));

  v3_prime_x = (double*)malloc(N*sizeof(double));  
  v3_prime_y = (double*)malloc(N*sizeof(double));
  v3_prime_z = (double*)malloc(N*sizeof(double));
  times = (int*)malloc(N*sizeof(int));

  cudaMalloc(&d_p1_x, N*sizeof(double)); 
  cudaMalloc(&d_p1_y, N*sizeof(double)); 
  cudaMalloc(&d_p1_z, N*sizeof(double)); 

  cudaMalloc(&d_p2_x, N*sizeof(double));
  cudaMalloc(&d_p2_y, N*sizeof(double));
  cudaMalloc(&d_p2_z, N*sizeof(double));

  cudaMalloc(&d_p3_x, N*sizeof(double));
  cudaMalloc(&d_p3_y, N*sizeof(double));
  cudaMalloc(&d_p3_z, N*sizeof(double));

  cudaMalloc(&d_p1_prime_x, N*sizeof(double));
  cudaMalloc(&d_p1_prime_y, N*sizeof(double));
  cudaMalloc(&d_p1_prime_z, N*sizeof(double));

  cudaMalloc(&d_p2_prime_x, N*sizeof(double));
  cudaMalloc(&d_p2_prime_y, N*sizeof(double));
  cudaMalloc(&d_p2_prime_z, N*sizeof(double));

  cudaMalloc(&d_p3_prime_x, N*sizeof(double));
  cudaMalloc(&d_p3_prime_y, N*sizeof(double));
  cudaMalloc(&d_p3_prime_z, N*sizeof(double));

  cudaMalloc(&d_v1_x, N*sizeof(double));
  cudaMalloc(&d_v1_y, N*sizeof(double));
  cudaMalloc(&d_v1_z, N*sizeof(double));

  cudaMalloc(&d_v2_x, N*sizeof(double));
  cudaMalloc(&d_v2_y, N*sizeof(double));
  cudaMalloc(&d_v2_z, N*sizeof(double));

  cudaMalloc(&d_v3_x, N*sizeof(double));
  cudaMalloc(&d_v3_y, N*sizeof(double));
  cudaMalloc(&d_v3_z, N*sizeof(double));

  cudaMalloc(&d_v1_prime_x, N*sizeof(double));
  cudaMalloc(&d_v1_prime_y, N*sizeof(double));
  cudaMalloc(&d_v1_prime_z, N*sizeof(double));

  cudaMalloc(&d_v2_prime_x, N*sizeof(double));
  cudaMalloc(&d_v2_prime_y, N*sizeof(double));
  cudaMalloc(&d_v2_prime_z, N*sizeof(double));

  cudaMalloc(&d_v3_prime_x, N*sizeof(double));
  cudaMalloc(&d_v3_prime_y, N*sizeof(double));
  cudaMalloc(&d_v3_prime_z, N*sizeof(double));
  cudaMalloc(&d_times, N*sizeof(int));

  int resolution = sqrt(N);
  double range = 40;
  double step_size = range / resolution;
  for (int i = 0; i < N; i++) {
    int remainder = i % resolution;
    int step = i / resolution;
    p1_x[i] = -20. + 40*(double(remainder)/double(resolution));
    p1_y[i] = -20. + 40*(double(step)/double(resolution));
    p1_z[i] = -11.0;

    p2_x[i] = 0.0;
    p2_y[i] = 0.0;
    p2_z[i] = 0.0;

    p3_x[i] = 10.0;
    p3_y[i] = 10.0;
    p3_z[i] = 12.0;

    // shift p1 in all x, y, z vectors
    p1_prime_x[i] = -20. + 40*(double(remainder)/double(resolution)) + 0.001;
    p1_prime_y[i] = -20. + 40*(double(step)/double(resolution)) + 0.001;
    p1_prime_z[i] = -11.0 + 0.001;

    p2_prime_x[i] = 0.0;
    p2_prime_y[i] = 0.0;
    p2_prime_z[i] = 0.0;

    p3_prime_x[i] = 10.0;
    p3_prime_y[i] = 10.0;
    p3_prime_z[i] = 12.0;

    v1_x[i] = -3.;
    v1_y[i] = 0.;
    v1_z[i] = 0.;

    v2_x[i] = 0.;
    v2_y[i] = 0.;
    v2_z[i] = 0.;

    v3_x[i] = 3.;
    v3_y[i] = 0.;
    v3_z[i] = 0.;

    v1_prime_x[i] = -3.;
    v1_prime_y[i] = 0.;
    v1_prime_z[i] = 0.;

    v2_prime_x[i] = 0.;
    v2_prime_y[i] = 0.;
    v2_prime_z[i] = 0.;

    v3_prime_x[i] = 3.;
    v3_prime_y[i] = 0.;
    v3_prime_z[i] = 0.;

    times[i] = 0;
}

  cudaMemcpy(d_p1_x, p1_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p1_y, p1_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p1_z, p1_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_p2_x, p2_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p2_y, p2_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p2_z, p2_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_p3_x, p3_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p3_y, p3_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p3_z, p3_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_p1_prime_x, p1_prime_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p1_prime_y, p1_prime_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p1_prime_z, p1_prime_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_p2_prime_x, p2_prime_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p2_prime_y, p2_prime_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p2_prime_z, p2_prime_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_p3_prime_x, p3_prime_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p3_prime_y, p3_prime_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p3_prime_z, p3_prime_z, N*sizeof(double), cudaMemcpyHostToDevice);


  cudaMemcpy(d_v1_x, v1_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v1_y, v1_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v1_z, v1_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_v2_x, v2_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v2_y, v2_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v2_z, v2_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_v3_x, v3_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v3_y, v3_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v3_z, v3_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_v1_prime_x, v1_prime_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v1_prime_y, v1_prime_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v1_prime_z, v1_prime_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_v2_prime_x, v2_prime_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v2_prime_y, v2_prime_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v2_prime_z, v2_prime_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_v3_prime_x, v3_prime_x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v3_prime_y, v3_prime_y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v3_prime_z, v3_prime_z, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_times, times, N*sizeof(int), cudaMemcpyHostToDevice);
  std::chrono::time_point<std::chrono::system_clock> start, end;
 
  start = std::chrono::system_clock::now();
  // call CUDA kernal in <<< , >>>>
  divergence<<<(N+127)/128, 128>>>(
      N, 
      steps, 
      delta_t,
      d_times,
      m1, m2, m3,
      critical_distance,
      d_p1_x, d_p1_y, d_p1_z, 
      d_p2_x, d_p2_y, d_p2_z, 
      d_p3_x, d_p3_y, d_p3_z, 
      d_p1_prime_x, d_p1_prime_y, d_p1_prime_z, 
      d_p2_prime_x, d_p2_prime_y, d_p2_prime_z, 
      d_p3_prime_x, d_p3_prime_y, d_p3_prime_z,
      d_v1_x, d_v1_y, d_v1_z,
      d_v2_x, d_v2_y, d_v2_z,
      d_v3_x, d_v3_y, d_v3_z,
      d_v1_prime_x, d_v1_prime_y, d_v1_prime_z,
      d_v2_prime_x, d_v2_prime_y, d_v2_prime_z,
      d_v3_prime_x, d_v3_prime_y, d_v3_prime_z);

  cudaDeviceSynchronize();

  // measure elapsed kernal runtime
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "Elapsed Time: " << elapsed_seconds.count() << "s\n";

  cudaMemcpy(times, d_times, N*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(p1_x, d_p1_x, N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(p1_y, d_p1_y, N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(p1_z, d_p1_z, N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(p1_prime_x, d_p1_prime_x, N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(p1_prime_y, d_p1_prime_y, N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(p1_prime_z, d_p1_prime_z, N*sizeof(double), cudaMemcpyDeviceToHost);
  for (int k=0; k<3; k++) {
    std::cout << times[k] << '\n';
    std::cout << p1_x[k] << ' ';
    std::cout << p1_y[k] << ' ';
    std::cout << p1_z[k] << ' ';
    std::cout << p1_prime_x[k] << ' ';
    std::cout << p1_prime_y[k] << ' ';
    std::cout << p1_prime_z[k] << ' ';
    std::cout << '\n';
  }


  cudaFree(d_p1_x); cudaFree(d_p1_y); cudaFree(d_p1_z);
  cudaFree(d_p2_x); cudaFree(d_p2_y); cudaFree(d_p2_z);
  cudaFree(d_p3_x); cudaFree(d_p3_y); cudaFree(d_p3_z);

  cudaFree(d_p1_prime_x); cudaFree(d_p1_prime_y); cudaFree(d_p1_prime_z);  
  cudaFree(d_p2_prime_x); cudaFree(d_p2_prime_y); cudaFree(d_p2_prime_z);
  cudaFree(d_p3_prime_x); cudaFree(d_p3_prime_y); cudaFree(d_p3_prime_z);

  cudaFree(d_v1_x); cudaFree(d_v1_y); cudaFree(d_v1_z);
  cudaFree(d_v2_x); cudaFree(d_v2_y); cudaFree(d_v2_z);
  cudaFree(d_v3_x); cudaFree(d_v3_y); cudaFree(d_v3_z);

  cudaFree(d_v1_prime_x); cudaFree(d_v1_prime_y); cudaFree(d_v1_prime_z);  
  cudaFree(d_v2_prime_x); cudaFree(d_v2_prime_y); cudaFree(d_v2_prime_z);
  cudaFree(d_v3_prime_x); cudaFree(d_v3_prime_y); cudaFree(d_v3_prime_z);

  cudaFree(d_times);

  free(p1_x); free(p1_y); free(p1_z);
  free(p2_x); free(p2_y); free(p2_z);
  free(p3_x); free(p3_y); free(p3_z);

  free(p1_prime_x); free(p1_prime_y); free(p1_prime_z);  
  free(p2_prime_x); free(p2_prime_y); free(p2_prime_z);  
  free(p3_prime_x); free(p3_prime_y); free(p3_prime_z);  

  free(v1_x); free(v1_y); free(v1_z);
  free(v2_x); free(v2_y); free(v2_z);
  free(v3_x); free(v3_y); free(v3_z);

  free(v1_prime_x); free(v1_prime_y); free(v1_prime_z);  
  free(v2_prime_x); free(v2_prime_y); free(v2_prime_z);  
  free(v3_prime_x); free(v3_prime_y); free(v3_prime_z);  

  free(times);
}










