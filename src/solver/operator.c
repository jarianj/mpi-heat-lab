#include "heat.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double medium_value(const Params *p, const double x[AXES]) {
    if (p->medium == MEDIUM_LAYERS) return x[0] < 0.5 ? p->alpha : p->alpha2;
    if (p->medium == MEDIUM_INCLUSION) {
        double dx = x[0] - 0.5, dy = x[1] - 0.5, dz = p->n[2] > 1 ? x[2] - 0.5 : 0.0;
        return dx * dx + dy * dy + dz * dz < 0.04 ? p->alpha2 : p->alpha;
    }
    return p->alpha;
}

static double harmonic(double a, double b) { return a + b > 0 ? 2.0 * a * b / (a + b) : 0.0; }

static void build_faces(Operator *op, const double *alpha) {
    const Grid *g = op->grid;
    for (int a = 0; a < AXES; a++) {
        op->face[a] = calloc(g->cells, sizeof(double));
        if (!op->face[a]) MPI_Abort(g->comm, 3);
        if (!g->active[a]) continue;
        int end[AXES] = {g->alloc[0], g->alloc[1], g->alloc[2]};
        end[a] -= 1;
        for (int k = 0; k < end[2]; k++) {
            for (int j = 0; j < end[1]; j++) {
                for (int i = 0; i < end[0]; i++) {
                    ptrdiff_t q = grid_index(g, i, j, k);
                    op->face[a][q] = harmonic(alpha[q], alpha[q + g->stride[a]]) * g->inv_h2[a];
                }
            }
        }
    }
}

int operator_create(Operator *op, const Grid *g, const Params *p, char *err) {
    op->grid = g;
    for (int a = 0; a < AXES; a++) {
        op->face[a] = NULL;
        op->coeff[a] = p->alpha * g->inv_h2[a];
    }
    op->alpha_max = p->alpha;
    if (p->medium == MEDIUM_UNIFORM) return 0;

    double *alpha = field_alloc(g);
    int lo[AXES], hi[AXES];
    grid_interior(g, lo, hi);
    if (p->medium == MEDIUM_FILE) {
        if (field_read(g, p->alpha_file, alpha, err)) {
            free(alpha);
            return 1;
        }
    } else {
        for (int k = lo[2]; k < hi[2]; k++) {
            for (int j = lo[1]; j < hi[1]; j++) {
                for (int i = lo[0]; i < hi[0]; i++) {
                    double x[AXES];
                    grid_position(g, i, j, k, x);
                    alpha[grid_index(g, i, j, k)] = medium_value(p, x);
                }
            }
        }
    }

    double local_max = 0.0, local_min = HUGE_VAL;
    for (int k = lo[2]; k < hi[2]; k++) {
        for (int j = lo[1]; j < hi[1]; j++) {
            for (int i = lo[0]; i < hi[0]; i++) {
                double value = alpha[grid_index(g, i, j, k)];
                if (value > local_max) local_max = value;
                if (value < local_min) local_min = value;
            }
        }
    }
    double global_min;
    MPI_Allreduce(&local_max, &op->alpha_max, 1, MPI_DOUBLE, MPI_MAX, g->comm);
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, g->comm);
    if (!(global_min > 0.0)) {
        snprintf(err, ERROR_LEN, "diffusivity field must be strictly positive");
        free(alpha);
        return 1;
    }

    Timers ignored = {0};
    fill_boundary(g, BC_NEUMANN, 0.0, 0.0, alpha, true);
    halo_exchange(g, p->halo == HALO_CHAIN ? HALO_SENDRECV : p->halo, alpha, &ignored);
    build_faces(op, alpha);
    free(alpha);
    return 0;
}

void operator_free(Operator *op) {
    for (int a = 0; a < AXES; a++) free(op->face[a]);
}

static void update_uniform(const Operator *op, double scale, const double *restrict u,
                           double *restrict out, const int lo[AXES], const int hi[AXES]) {
    const Grid *g = op->grid;
    const ptrdiff_t sx = g->stride[0], sy = g->stride[1], sz = g->stride[2];
    const double cx = scale * op->coeff[0], cy = scale * op->coeff[1], cz = scale * op->coeff[2];
    const double center = 1.0 - 2.0 * (cx + cy + cz);
    const int count = hi[0] - lo[0];
    for (int k = lo[2]; k < hi[2]; k++) {
        for (int j = lo[1]; j < hi[1]; j++) {
            const ptrdiff_t base = grid_index(g, lo[0], j, k);
            const double *restrict v = u + base;
            double *restrict w = out + base;
            for (int i = 0; i < count; i++) {
                w[i] = center * v[i] + cx * (v[i - sx] + v[i + sx]) + cy * (v[i - sy] + v[i + sy]) +
                       cz * (v[i - sz] + v[i + sz]);
            }
        }
    }
}

static void update_variable(const Operator *op, double scale, const double *restrict u,
                            double *restrict out, const int lo[AXES], const int hi[AXES]) {
    const Grid *g = op->grid;
    const ptrdiff_t sx = g->stride[0], sy = g->stride[1], sz = g->stride[2];
    const double *restrict fx = op->face[0], *restrict fy = op->face[1], *restrict fz = op->face[2];
    for (int k = lo[2]; k < hi[2]; k++) {
        for (int j = lo[1]; j < hi[1]; j++) {
            for (int i = lo[0]; i < hi[0]; i++) {
                const ptrdiff_t q = grid_index(g, i, j, k);
                const double c = u[q];
                const double flux = fx[q] * (u[q + sx] - c) - fx[q - sx] * (c - u[q - sx]) +
                                    fy[q] * (u[q + sy] - c) - fy[q - sy] * (c - u[q - sy]) +
                                    fz[q] * (u[q + sz] - c) - fz[q - sz] * (c - u[q - sz]);
                out[q] = c + scale * flux;
            }
        }
    }
}

void operator_update(const Operator *op, double scale, const double *u, double *out,
                     const int lo[AXES], const int hi[AXES]) {
    for (int a = 0; a < AXES; a++) {
        if (lo[a] >= hi[a]) return;
    }
    if (op->face[0]) {
        update_variable(op, scale, u, out, lo, hi);
    } else {
        update_uniform(op, scale, u, out, lo, hi);
    }
}

void operator_update_all(const Operator *op, double scale, const double *u, double *out) {
    int lo[AXES], hi[AXES];
    grid_interior(op->grid, lo, hi);
    operator_update(op, scale, u, out, lo, hi);
}

void operator_update_inner(const Operator *op, double scale, const double *u, double *out) {
    const Grid *g = op->grid;
    int lo[AXES], hi[AXES];
    grid_interior(g, lo, hi);
    for (int a = 0; a < AXES; a++) {
        if (g->active[a]) {
            lo[a] += 1;
            hi[a] -= 1;
        }
    }
    operator_update(op, scale, u, out, lo, hi);
}

void operator_update_shell(const Operator *op, double scale, const double *u, double *out) {
    const Grid *g = op->grid;
    int lo[AXES], hi[AXES];
    grid_interior(g, lo, hi);
    for (int a = 0; a < AXES; a++) {
        if (!g->active[a] || lo[a] >= hi[a]) continue;
        int slab_lo[AXES] = {lo[0], lo[1], lo[2]}, slab_hi[AXES] = {hi[0], hi[1], hi[2]};
        slab_hi[a] = lo[a] + 1;
        operator_update(op, scale, u, out, slab_lo, slab_hi);
        if (hi[a] - lo[a] > 1) {
            slab_lo[a] = hi[a] - 1;
            slab_hi[a] = hi[a];
            operator_update(op, scale, u, out, slab_lo, slab_hi);
        }
        lo[a] += 1;
        hi[a] -= 1;
    }
}

static double interior_dot(const Grid *g, const double *x, const double *y, Timers *t) {
    double start = MPI_Wtime();
    int lo[AXES], hi[AXES];
    grid_interior(g, lo, hi);
    double local = 0.0;
    for (int k = lo[2]; k < hi[2]; k++) {
        for (int j = lo[1]; j < hi[1]; j++) {
            for (int i = lo[0]; i < hi[0]; i++) {
                ptrdiff_t q = grid_index(g, i, j, k);
                local += x[q] * y[q];
            }
        }
    }
    double middle = MPI_Wtime();
    double global;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, g->comm);
    t->compute += middle - start;
    t->reduce += MPI_Wtime() - middle;
    return global;
}

void cg_create(CgWorkspace *w, const Operator *op, double scale) {
    const Grid *g = op->grid;
    w->r = field_alloc(g);
    w->z = field_alloc(g);
    w->p = field_alloc(g);
    w->q = field_alloc(g);
    w->diag = field_alloc(g);
    int lo[AXES], hi[AXES];
    grid_interior(g, lo, hi);
    const double uniform = 2.0 * (op->coeff[0] + op->coeff[1] + op->coeff[2]);
    for (int k = lo[2]; k < hi[2]; k++) {
        for (int j = lo[1]; j < hi[1]; j++) {
            for (int i = lo[0]; i < hi[0]; i++) {
                ptrdiff_t q = grid_index(g, i, j, k);
                double sum = uniform;
                if (op->face[0]) {
                    sum = 0.0;
                    for (int a = 0; a < AXES; a++) {
                        sum += op->face[a][q] + op->face[a][q - g->stride[a]];
                    }
                }
                w->diag[q] = 1.0 - scale * sum;
            }
        }
    }
}

void cg_free(CgWorkspace *w) {
    free(w->r);
    free(w->z);
    free(w->p);
    free(w->q);
    free(w->diag);
}

static void matvec(const Operator *op, const Params *p, double scale, double *v, double *out,
                   Timers *t) {
    const Grid *g = op->grid;
    double start = MPI_Wtime();
    fill_boundary(g, p->boundary, 0.0, 0.0, v, true);
    t->compute += MPI_Wtime() - start;
    halo_exchange(g, p->halo, v, t);
    start = MPI_Wtime();
    operator_update_all(op, scale, v, out);
    t->compute += MPI_Wtime() - start;
}

int cg_solve(const Operator *op, const Params *p, CgWorkspace *w, double scale, const double *b,
             double *x, Timers *t) {
    const Grid *g = op->grid;
    int lo[AXES], hi[AXES];
    grid_interior(g, lo, hi);

    matvec(op, p, scale, x, w->q, t);
    double start = MPI_Wtime();
    for (int k = lo[2]; k < hi[2]; k++) {
        for (int j = lo[1]; j < hi[1]; j++) {
            for (int i = lo[0]; i < hi[0]; i++) {
                ptrdiff_t q = grid_index(g, i, j, k);
                w->r[q] = b[q] - w->q[q];
                w->z[q] = w->r[q] / w->diag[q];
                w->p[q] = w->z[q];
            }
        }
    }
    t->compute += MPI_Wtime() - start;

    double b_norm2 = interior_dot(g, b, b, t);
    double threshold = p->cg_tol * p->cg_tol * (b_norm2 > 0.0 ? b_norm2 : 1.0);
    double rz = interior_dot(g, w->r, w->z, t);
    double r2 = interior_dot(g, w->r, w->r, t);

    int iterations = 0;
    while (r2 > threshold && iterations < p->cg_max) {
        matvec(op, p, scale, w->p, w->q, t);
        double step = rz / interior_dot(g, w->p, w->q, t);
        start = MPI_Wtime();
        for (int k = lo[2]; k < hi[2]; k++) {
            for (int j = lo[1]; j < hi[1]; j++) {
                for (int i = lo[0]; i < hi[0]; i++) {
                    ptrdiff_t q = grid_index(g, i, j, k);
                    x[q] += step * w->p[q];
                    w->r[q] -= step * w->q[q];
                    w->z[q] = w->r[q] / w->diag[q];
                }
            }
        }
        t->compute += MPI_Wtime() - start;
        double rz_next = interior_dot(g, w->r, w->z, t);
        r2 = interior_dot(g, w->r, w->r, t);
        double beta = rz_next / rz;
        rz = rz_next;
        start = MPI_Wtime();
        for (int k = lo[2]; k < hi[2]; k++) {
            for (int j = lo[1]; j < hi[1]; j++) {
                for (int i = lo[0]; i < hi[0]; i++) {
                    ptrdiff_t q = grid_index(g, i, j, k);
                    w->p[q] = w->z[q] + beta * w->p[q];
                }
            }
        }
        t->compute += MPI_Wtime() - start;
        iterations++;
    }
    return r2 <= threshold ? iterations : -iterations - 1;
}
