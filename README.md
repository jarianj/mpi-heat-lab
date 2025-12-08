# 🧊 Parallel 1-D Heat Diffusion Solver (MPI)

A high-performance parallel implementation of the 1-dimensional heat diffusion equation using MPI domain decomposition and ghost-cell boundary exchange.

This project demonstrates how large-scale numerical simulations can be efficiently distributed across multiple processes using the Message Passing Interface (MPI).

## 📘 About the Project

This program solves the heat equation:

$$\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2}$$

using:

- **Finite Difference Method (FDM)** for spatial discretization
- **Explicit Euler** for time integration
- **MPI parallelization** with domain decomposition
- **Ghost cells** for neighbor communication
- **Neumann boundary conditions** (zero flux at global boundaries)

It is designed to handle very large grids (hundreds of millions of cells) by dividing the work across multiple MPI processes.

## 🧠 Mathematical Background

The continuous equation:

$$\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2}$$

is discretized as:

$$T_i^{k+1} = T_i^k + \Delta t \cdot \alpha (T_{i-1}^k - 2T_i^k + T_{i+1}^k)$$

**Key concepts:**

- **Curvature term** $T_{i-1} - 2T_i + T_{i+1}$ controls how heat flows
- **Positive curvature**: point warms up
- **Negative curvature**: point cools down
- **Boundary**: insulated (no heat leaves the system)

## ⚙️ Parallelization Strategy (MPI)

### 🔹 Domain Decomposition

The global grid is divided evenly:

```
[ Process 0 ] [ Process 1 ] [ Process 2 ] ... [ Process N ]
```

Each process stores:

- Its own block of real cells: `1 ... block_size`
- Two ghost cells: `0` (left) and `block_size+1` (right)

### 🔹 Ghost Cell Exchange

Every time step:

1. Receive left neighbor's boundary → store in left ghost
2. Send my first real cell to left
3. Receive right neighbor's boundary → store in right ghost
4. Send my last real cell to right

This ensures every process has the correct neighbor values to update its interior.

### 🔹 Load Balance

All processes do equal work → ideal for strong scaling.

## 🏗️ Project Structure

```
.
├── main.c              # Main MPI heat equation solver
├── mpicompile.ps1      # PowerShell compilation script (Windows)
├── mpirun.ps1          # PowerShell execution script (Windows)
├── README.md           # This file
└── .gitignore          # Git ignore rules
```

## 🧪 How to Build

### Windows (PowerShell)

Using the provided script:

```powershell
.\mpicompile.ps1
```

Or compile manually with MS-MPI:

```powershell
gcc -I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" main.c -L"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -lmsmpi -o main.exe
```

### Linux/Unix

Compile with `mpicc`:

```bash
mpicc -O3 -Wall -o main main.c
```

## ▶️ How to Run

### Windows (PowerShell)

Using the provided script:

```powershell
.\mpirun.ps1 -Processes 4
```

Or run manually:

```powershell
mpiexec -n 4 main.exe
```

### Linux/Unix

```bash
mpirun -np 4 ./main
```

**Output:**

```
Time step 0 of 100
T_average: 1.000000
```

## 📈 Performance Notes

- Communication is done via `MPI_Sendrecv` to avoid deadlock
- Using ghost cells avoids conditional logic inside compute loops
- Communication pattern is deadlock-free
- **Complexity per time step:**
  - Compute: $O(N/P)$
  - Communication: constant (2 neighbors)
- Easily scalable to millions of grid points

## 📊 Result Interpretation

The printed average temperature:

```
T_average = <value>
```

Confirms that the system conserves energy (Neumann boundaries).

## 🚀 Future Improvements

- Switch to `MPI_Isend`/`MPI_Irecv` for communication-computation overlap
- Extend to 2-D/3-D heat diffusion
- Implement stability checks for large $\Delta t$
- Add visualization scripts (Python / Matplotlib)
- Add command-line arguments for grid size, time steps, etc.

## 📝 License

This project is open source and available for educational purposes.
