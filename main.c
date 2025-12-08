#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * Parallel 1D Heat Equation Solver using MPI
 * 
 * This program solves the 1D heat equation using finite difference method
 * with parallelization across multiple MPI processes.
 */

int main(int argc, char** argv) {
    int rank, num_procs;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    // Simulation parameters
    int grid_size = 1024;  // Total number of grid points (reduced for testing)
    int block_size = grid_size / num_procs;
    double delta_t = 0.02;  // Time step size
    int num_time_steps = 100;  // Number of time steps (reduced for testing)
    double conductivity = 0.1;  // Thermal conductivity coefficient
    
    // Allocate memory for local arrays (with ghost cells)
    int local_size = block_size + 2;  // +2 for ghost cells on each side
    double* T_k = (double*)malloc(sizeof(double) * local_size);
    double* T_kn = (double*)malloc(sizeof(double) * local_size);
    
    if (T_k == NULL || T_kn == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Finalize();
        return 1;
    }
    
    // Initialize temperature arrays
    for(int i = 0; i < local_size; i++) {
        T_k[i] = 1.0;  // Initial temperature
        T_kn[i] = 1.0;
    }
    
    // Time stepping loop
    for(int k = 0; k < num_time_steps; k++) {
        // Print progress every 100 steps
        if(rank == 0 && k % 100 == 0) {
            printf("Time step %d of %d\n", k, num_time_steps);
        }
        
        // Exchange ghost cells with neighbors using MPI_Sendrecv to avoid deadlock
        if(rank > 0 && rank < num_procs - 1) {
            // Middle processes: exchange with both neighbors
            MPI_Sendrecv(&T_k[1], 1, MPI_DOUBLE, rank - 1, 0,
                        &T_k[0], 1, MPI_DOUBLE, rank - 1, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&T_k[block_size], 1, MPI_DOUBLE, rank + 1, 0,
                        &T_k[block_size + 1], 1, MPI_DOUBLE, rank + 1, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if(rank > 0) {
            // Last process: only exchange with left neighbor
            MPI_Sendrecv(&T_k[1], 1, MPI_DOUBLE, rank - 1, 0,
                        &T_k[0], 1, MPI_DOUBLE, rank - 1, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if(rank < num_procs - 1) {
            // First process: only exchange with right neighbor
            MPI_Sendrecv(&T_k[block_size], 1, MPI_DOUBLE, rank + 1, 0,
                        &T_k[block_size + 1], 1, MPI_DOUBLE, rank + 1, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        // Handle boundary conditions for first and last ranks
        if(rank == 0) {
            // Left boundary: reflecting boundary condition
            T_k[0] = T_k[1];
        }
        if(rank == num_procs - 1) {
            // Right boundary: reflecting boundary condition
            T_k[block_size + 1] = T_k[block_size];
        }
        
        // Update interior cells using finite difference method
        for(int i = 1; i <= block_size; i++) {
            double dTdt = conductivity * (T_k[i - 1] - 2 * T_k[i] + T_k[i + 1]);
            T_kn[i] = T_k[i] + delta_t * dTdt;
        }
        
        // Swap arrays for next iteration
        double* temp = T_k;
        T_k = T_kn;
        T_kn = temp;
    }
    
    // Compute global average temperature
    double local_sum = 0.0;
    for(int i = 1; i <= block_size; i++) {
        local_sum += T_k[i];
    }
    
    double global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if(rank == 0) {
        double T_average = global_sum / (double)grid_size;
        printf("T_average: %f\n", T_average);
    }
    
    // Free allocated memory
    free(T_k);
    free(T_kn);
    
    MPI_Finalize();
    
    return 0;
}