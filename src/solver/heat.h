#ifndef HEAT_H
#define HEAT_H

#include <mpi.h>
#include <stdbool.h>
#include <stddef.h>

enum { AXES = 3, ERROR_LEN = 256, PATH_LEN = 1024 };

typedef enum { BC_NEUMANN, BC_DIRICHLET, BC_PERIODIC } Boundary;
typedef enum { SCHEME_EXPLICIT, SCHEME_CRANK_NICOLSON, SCHEME_BACKWARD_EULER } Scheme;
typedef enum { HALO_CHAIN, HALO_SENDRECV, HALO_OVERLAP } HaloMode;
typedef enum { IC_UNIFORM, IC_GAUSSIAN, IC_STEP, IC_MODE, IC_FILE } InitialCondition;
typedef enum { MEDIUM_UNIFORM, MEDIUM_LAYERS, MEDIUM_INCLUSION, MEDIUM_FILE } Medium;

typedef struct {
    int n[AXES];
    int dims[AXES];
    int mode[AXES];
    Scheme scheme;
    HaloMode halo;
    Boundary boundary;
    InitialCondition ic;
    Medium medium;
    double alpha, alpha2;
    double bc_x0, bc_x1;
    double t_end, dt, cfl, cg_tol;
    int steps, cg_max, diag_every, snapshot_every;
    bool allow_unstable;
    char ic_file[PATH_LEN], alpha_file[PATH_LEN], output[PATH_LEN];
} Params;

typedef struct {
    MPI_Comm comm;
    int rank, size;
    int dims[AXES], coords[AXES], neighbor[AXES][2];
    int global[AXES], local[AXES], offset[AXES], ghost[AXES], alloc[AXES];
    ptrdiff_t stride[AXES];
    size_t cells;
    long long global_cells;
    double h[AXES], inv_h2[AXES], cell_volume;
    bool active[AXES];
    MPI_Datatype send[AXES][2], recv[AXES][2], interior_mem, interior_file;
} Grid;

typedef struct {
    const Grid *grid;
    double coeff[AXES];
    double *face[AXES];
    double alpha_max;
} Operator;

typedef struct {
    double compute, halo, reduce, io;
} Timers;

typedef struct {
    double *r, *z, *p, *q, *diag;
} CgWorkspace;

static inline ptrdiff_t grid_index(const Grid *g, int i, int j, int k) {
    return i + (ptrdiff_t)g->alloc[0] * (j + (ptrdiff_t)g->alloc[1] * k);
}

int params_parse(Params *p, int argc, char **argv, char *err);

int grid_create(Grid *g, const Params *p, MPI_Comm comm, char *err);
void grid_free(Grid *g);
void grid_interior(const Grid *g, int lo[AXES], int hi[AXES]);
void grid_position(const Grid *g, int i, int j, int k, double x[AXES]);
double *field_alloc(const Grid *g);

void halo_exchange(const Grid *g, HaloMode mode, double *f, Timers *t);
void halo_start(const Grid *g, double *f, MPI_Request req[4 * AXES]);
void halo_finish(MPI_Request req[4 * AXES]);
void fill_boundary(const Grid *g, Boundary bc, double x0, double x1, double *f, bool homogeneous);

int operator_create(Operator *op, const Grid *g, const Params *p, char *err);
void operator_free(Operator *op);
void operator_update(const Operator *op, double scale, const double *u, double *out,
                     const int lo[AXES], const int hi[AXES]);
void operator_update_all(const Operator *op, double scale, const double *u, double *out);
void operator_update_inner(const Operator *op, double scale, const double *u, double *out);
void operator_update_shell(const Operator *op, double scale, const double *u, double *out);

void cg_create(CgWorkspace *w, const Operator *op, double scale);
void cg_free(CgWorkspace *w);
int cg_solve(const Operator *op, const Params *p, CgWorkspace *w, double scale, const double *b,
             double *x, Timers *t);

int field_read(const Grid *g, const char *path, double *f, char *err);
int field_write(const Grid *g, const char *path, const double *f);

#endif
