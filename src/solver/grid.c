#include "heat.h"

#include <stdio.h>
#include <stdlib.h>

static void face_type(const Grid *g, int axis, int start, MPI_Datatype *type) {
    int sizes[AXES], subsizes[AXES], starts[AXES];
    for (int a = 0; a < AXES; a++) {
        int c = AXES - 1 - a;
        sizes[c] = g->alloc[a];
        subsizes[c] = a == axis ? 1 : g->local[a];
        starts[c] = a == axis ? start : g->ghost[a];
    }
    MPI_Type_create_subarray(AXES, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, type);
    MPI_Type_commit(type);
}

static void interior_types(Grid *g) {
    int alloc[AXES], local[AXES], global[AXES], ghost[AXES], offset[AXES];
    for (int a = 0; a < AXES; a++) {
        int c = AXES - 1 - a;
        alloc[c] = g->alloc[a];
        local[c] = g->local[a];
        global[c] = g->global[a];
        ghost[c] = g->ghost[a];
        offset[c] = g->offset[a];
    }
    MPI_Type_create_subarray(AXES, alloc, local, ghost, MPI_ORDER_C, MPI_DOUBLE, &g->interior_mem);
    MPI_Type_commit(&g->interior_mem);
    MPI_Type_create_subarray(AXES, global, local, offset, MPI_ORDER_C, MPI_DOUBLE, &g->interior_file);
    MPI_Type_commit(&g->interior_file);
}

int grid_create(Grid *g, const Params *p, MPI_Comm comm, char *err) {
    MPI_Comm_size(comm, &g->size);

    int dims[AXES], periods[AXES];
    int active_axes = 0;
    for (int a = 0; a < AXES; a++) {
        g->global[a] = p->n[a];
        g->active[a] = p->n[a] > 1;
        active_axes += g->active[a];
        dims[a] = g->active[a] ? p->dims[a] : 1;
        periods[a] = g->active[a] && p->boundary == BC_PERIODIC;
    }
    if (active_axes == 0) {
        snprintf(err, ERROR_LEN, "at least one grid axis must have more than one cell");
        return 1;
    }
    bool explicit_dims = p->dims[0] > 0 || p->dims[1] > 0 || p->dims[2] > 0;
    if (explicit_dims) {
        int product = 1;
        for (int a = 0; a < AXES; a++) {
            int requested = p->dims[a] > 0 ? p->dims[a] : 1;
            if (!g->active[a] && requested > 1) {
                snprintf(err, ERROR_LEN, "cannot decompose an axis with one cell");
                return 1;
            }
            dims[a] = requested;
            product *= requested;
        }
        if (product != g->size) {
            snprintf(err, ERROR_LEN, "--dims product %d does not match %d ranks", product, g->size);
            return 1;
        }
    } else {
        for (int a = 0; a < AXES; a++) dims[a] = g->active[a] ? 0 : 1;
        MPI_Dims_create(g->size, AXES, dims);
    }
    for (int a = 0; a < AXES; a++) {
        if (dims[a] > p->n[a]) {
            snprintf(err, ERROR_LEN, "axis %d has %d cells for %d ranks", a, p->n[a], dims[a]);
            return 1;
        }
    }

    MPI_Cart_create(comm, AXES, dims, periods, 0, &g->comm);
    MPI_Comm_rank(g->comm, &g->rank);
    MPI_Cart_coords(g->comm, g->rank, AXES, g->coords);

    g->global_cells = 1;
    g->cell_volume = 1.0;
    g->cells = 1;
    for (int a = 0; a < AXES; a++) {
        g->dims[a] = dims[a];
        MPI_Cart_shift(g->comm, a, 1, &g->neighbor[a][0], &g->neighbor[a][1]);
        int base = g->global[a] / dims[a], extra = g->global[a] % dims[a], c = g->coords[a];
        g->local[a] = base + (c < extra);
        g->offset[a] = c * base + (c < extra ? c : extra);
        g->ghost[a] = g->active[a] ? 1 : 0;
        g->alloc[a] = g->local[a] + 2 * g->ghost[a];
        g->cells *= (size_t)g->alloc[a];
        g->global_cells *= g->global[a];
        g->h[a] = 1.0 / g->global[a];
        g->inv_h2[a] = g->active[a] ? 1.0 / (g->h[a] * g->h[a]) : 0.0;
        if (g->active[a]) g->cell_volume *= g->h[a];
    }
    g->stride[0] = g->active[0] ? 1 : 0;
    g->stride[1] = g->active[1] ? g->alloc[0] : 0;
    g->stride[2] = g->active[2] ? (ptrdiff_t)g->alloc[0] * g->alloc[1] : 0;

    for (int a = 0; a < AXES; a++) {
        g->send[a][0] = g->send[a][1] = g->recv[a][0] = g->recv[a][1] = MPI_DATATYPE_NULL;
        if (!g->active[a]) continue;
        face_type(g, a, g->ghost[a], &g->send[a][0]);
        face_type(g, a, g->ghost[a] + g->local[a] - 1, &g->send[a][1]);
        face_type(g, a, 0, &g->recv[a][0]);
        face_type(g, a, g->ghost[a] + g->local[a], &g->recv[a][1]);
    }
    interior_types(g);
    return 0;
}

void grid_free(Grid *g) {
    for (int a = 0; a < AXES; a++) {
        if (!g->active[a]) continue;
        for (int s = 0; s < 2; s++) {
            MPI_Type_free(&g->send[a][s]);
            MPI_Type_free(&g->recv[a][s]);
        }
    }
    MPI_Type_free(&g->interior_mem);
    MPI_Type_free(&g->interior_file);
    MPI_Comm_free(&g->comm);
}

void grid_interior(const Grid *g, int lo[AXES], int hi[AXES]) {
    for (int a = 0; a < AXES; a++) {
        lo[a] = g->ghost[a];
        hi[a] = g->ghost[a] + g->local[a];
    }
}

void grid_position(const Grid *g, int i, int j, int k, double x[AXES]) {
    int index[AXES] = {i, j, k};
    for (int a = 0; a < AXES; a++) {
        x[a] = (index[a] - g->ghost[a] + g->offset[a] + 0.5) * g->h[a];
    }
}

double *field_alloc(const Grid *g) {
    double *f = calloc(g->cells, sizeof *f);
    if (!f) {
        fprintf(stderr, "error: rank %d could not allocate %zu cells\n", g->rank, g->cells);
        MPI_Abort(g->comm, 3);
    }
    return f;
}
