#include "heat.h"

static int tag(int axis, int direction) { return 2 * axis + direction; }

void halo_start(const Grid *g, double *f, MPI_Request req[4 * AXES]) {
    int n = 0;
    for (int a = 0; a < AXES; a++) {
        if (!g->active[a]) {
            for (int s = 0; s < 4; s++) req[n++] = MPI_REQUEST_NULL;
            continue;
        }
        MPI_Irecv(f, 1, g->recv[a][0], g->neighbor[a][0], tag(a, 1), g->comm, &req[n++]);
        MPI_Irecv(f, 1, g->recv[a][1], g->neighbor[a][1], tag(a, 0), g->comm, &req[n++]);
        MPI_Isend(f, 1, g->send[a][0], g->neighbor[a][0], tag(a, 0), g->comm, &req[n++]);
        MPI_Isend(f, 1, g->send[a][1], g->neighbor[a][1], tag(a, 1), g->comm, &req[n++]);
    }
}

void halo_finish(MPI_Request req[4 * AXES]) {
    MPI_Status statuses[4 * AXES];
    MPI_Waitall(4 * AXES, req, statuses);
}

static void exchange_chain(const Grid *g, double *f) {
    for (int a = 0; a < AXES; a++) {
        if (!g->active[a]) continue;
        MPI_Sendrecv(f, 1, g->send[a][0], g->neighbor[a][0], tag(a, 0), f, 1, g->recv[a][0],
                     g->neighbor[a][0], tag(a, 1), g->comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(f, 1, g->send[a][1], g->neighbor[a][1], tag(a, 1), f, 1, g->recv[a][1],
                     g->neighbor[a][1], tag(a, 0), g->comm, MPI_STATUS_IGNORE);
    }
}

static void exchange_sendrecv(const Grid *g, double *f) {
    for (int a = 0; a < AXES; a++) {
        if (!g->active[a]) continue;
        MPI_Sendrecv(f, 1, g->send[a][0], g->neighbor[a][0], tag(a, 0), f, 1, g->recv[a][1],
                     g->neighbor[a][1], tag(a, 0), g->comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(f, 1, g->send[a][1], g->neighbor[a][1], tag(a, 1), f, 1, g->recv[a][0],
                     g->neighbor[a][0], tag(a, 1), g->comm, MPI_STATUS_IGNORE);
    }
}

void halo_exchange(const Grid *g, HaloMode mode, double *f, Timers *t) {
    double start = MPI_Wtime();
    if (mode == HALO_CHAIN) {
        exchange_chain(g, f);
    } else if (mode == HALO_SENDRECV) {
        exchange_sendrecv(g, f);
    } else {
        MPI_Request req[4 * AXES];
        halo_start(g, f, req);
        halo_finish(req);
    }
    t->halo += MPI_Wtime() - start;
}

static double ghost_value(Boundary bc, double inside, double wall, bool homogeneous) {
    if (bc == BC_NEUMANN) return inside;
    return (homogeneous ? 0.0 : 2.0 * wall) - inside;
}

void fill_boundary(const Grid *g, Boundary bc, double x0, double x1, double *f, bool homogeneous) {
    if (bc == BC_PERIODIC) return;
    int lo[AXES], hi[AXES];
    grid_interior(g, lo, hi);
    for (int a = 0; a < AXES; a++) {
        if (!g->active[a]) continue;
        int b = (a + 1) % AXES, c = (a + 2) % AXES;
        for (int side = 0; side < 2; side++) {
            if (g->neighbor[a][side] != MPI_PROC_NULL) continue;
            double wall = a == 0 ? (side == 0 ? x0 : x1) : 0.0;
            int ghost_at = side == 0 ? 0 : hi[a];
            int inside_at = side == 0 ? lo[a] : hi[a] - 1;
            int idx[AXES];
            for (int v = lo[c]; v < hi[c]; v++) {
                for (int u = lo[b]; u < hi[b]; u++) {
                    idx[b] = u;
                    idx[c] = v;
                    idx[a] = inside_at;
                    double inside = f[grid_index(g, idx[0], idx[1], idx[2])];
                    idx[a] = ghost_at;
                    f[grid_index(g, idx[0], idx[1], idx[2])] =
                        ghost_value(bc, inside, wall, homogeneous);
                }
            }
        }
    }
}
