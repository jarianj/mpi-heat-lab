#include "heat.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef _WIN32
#include <direct.h>
#define make_directory(path) _mkdir(path)
#else
#include <sys/stat.h>
#define make_directory(path) mkdir(path, 0777)
#endif

typedef struct {
    double energy, min, max;
} Stats;

static const char *SCHEME_NAMES[] = {"explicit", "cn", "be"};
static const char *HALO_NAMES[] = {"chain", "sendrecv", "overlap"};
static const char *BOUNDARY_NAMES[] = {"neumann", "dirichlet", "periodic"};

static double mode_rate(const Params *p, const Grid *g) {
    double wave = p->boundary == BC_PERIODIC ? 2.0 * M_PI : M_PI;
    double sum = 0.0;
    for (int a = 0; a < AXES; a++) {
        if (g->active[a]) sum += p->mode[a] * p->mode[a];
    }
    return p->alpha * wave * wave * sum;
}

static double mode_shape(const Params *p, const Grid *g, const double x[AXES]) {
    double value = 1.0;
    for (int a = 0; a < AXES; a++) {
        if (!g->active[a]) continue;
        double k = p->mode[a];
        if (p->boundary == BC_NEUMANN) value *= cos(k * M_PI * x[a]);
        else if (p->boundary == BC_DIRICHLET) value *= sin(k * M_PI * x[a]);
        else value *= cos(2.0 * k * M_PI * x[a]);
    }
    return value;
}

static double initial_value(const Params *p, const Grid *g, const double x[AXES]) {
    switch (p->ic) {
    case IC_UNIFORM: return 1.0;
    case IC_STEP: return x[0] < 0.5 ? 1.0 : 0.0;
    case IC_MODE: return mode_shape(p, g, x);
    default: break;
    }
    double r2 = 0.0;
    for (int a = 0; a < AXES; a++) {
        if (g->active[a]) r2 += (x[a] - 0.5) * (x[a] - 0.5);
    }
    return exp(-r2 / (2.0 * 0.1 * 0.1));
}

static int initialize(const Params *p, const Grid *g, double *u, char *err) {
    if (p->ic == IC_FILE) return field_read(g, p->ic_file, u, err);
    int lo[AXES], hi[AXES];
    grid_interior(g, lo, hi);
    for (int k = lo[2]; k < hi[2]; k++) {
        for (int j = lo[1]; j < hi[1]; j++) {
            for (int i = lo[0]; i < hi[0]; i++) {
                double x[AXES];
                grid_position(g, i, j, k, x);
                u[grid_index(g, i, j, k)] = initial_value(p, g, x);
            }
        }
    }
    return 0;
}

static Stats field_stats(const Grid *g, const double *u) {
    int lo[AXES], hi[AXES];
    grid_interior(g, lo, hi);
    double sum = 0.0, min = HUGE_VAL, max = -HUGE_VAL;
    for (int k = lo[2]; k < hi[2]; k++) {
        for (int j = lo[1]; j < hi[1]; j++) {
            for (int i = lo[0]; i < hi[0]; i++) {
                double value = u[grid_index(g, i, j, k)];
                sum += value;
                if (value < min) min = value;
                if (value > max) max = value;
            }
        }
    }
    Stats out;
    MPI_Allreduce(&sum, &out.energy, 1, MPI_DOUBLE, MPI_SUM, g->comm);
    MPI_Allreduce(&min, &out.min, 1, MPI_DOUBLE, MPI_MIN, g->comm);
    MPI_Allreduce(&max, &out.max, 1, MPI_DOUBLE, MPI_MAX, g->comm);
    out.energy *= g->cell_volume;
    return out;
}

static void analytic_error(const Params *p, const Grid *g, const double *u, double t, double *l2,
                           double *linf) {
    int lo[AXES], hi[AXES];
    grid_interior(g, lo, hi);
    double decay = exp(-mode_rate(p, g) * t);
    double sum = 0.0, peak = 0.0;
    for (int k = lo[2]; k < hi[2]; k++) {
        for (int j = lo[1]; j < hi[1]; j++) {
            for (int i = lo[0]; i < hi[0]; i++) {
                double x[AXES];
                grid_position(g, i, j, k, x);
                double e = fabs(u[grid_index(g, i, j, k)] - decay * mode_shape(p, g, x));
                sum += e * e;
                if (e > peak) peak = e;
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, g->comm);
    MPI_Allreduce(&peak, linf, 1, MPI_DOUBLE, MPI_MAX, g->comm);
    *l2 = sqrt(sum * g->cell_volume);
}

static void step_explicit(const Operator *op, const Params *p, double dt, double *u, double *next,
                          Timers *t) {
    const Grid *g = op->grid;
    double start = MPI_Wtime();
    fill_boundary(g, p->boundary, p->bc_x0, p->bc_x1, u, false);
    t->compute += MPI_Wtime() - start;
    if (p->halo == HALO_OVERLAP) {
        MPI_Request req[4 * AXES];
        start = MPI_Wtime();
        halo_start(g, u, req);
        t->halo += MPI_Wtime() - start;
        start = MPI_Wtime();
        operator_update_inner(op, dt, u, next);
        t->compute += MPI_Wtime() - start;
        start = MPI_Wtime();
        halo_finish(req);
        t->halo += MPI_Wtime() - start;
        start = MPI_Wtime();
        operator_update_shell(op, dt, u, next);
        t->compute += MPI_Wtime() - start;
    } else {
        halo_exchange(g, p->halo, u, t);
        start = MPI_Wtime();
        operator_update_all(op, dt, u, next);
        t->compute += MPI_Wtime() - start;
    }
}

static int step_implicit(const Operator *op, const Params *p, CgWorkspace *w, double dt, double theta,
                         const double *wall, double *u, double *next, double *rhs, Timers *t) {
    const Grid *g = op->grid;
    double start = MPI_Wtime();
    fill_boundary(g, p->boundary, 0.0, 0.0, u, true);
    t->compute += MPI_Wtime() - start;
    halo_exchange(g, p->halo, u, t);
    start = MPI_Wtime();
    operator_update_all(op, (1.0 - theta) * dt, u, rhs);
    int lo[AXES], hi[AXES];
    grid_interior(g, lo, hi);
    for (int k = lo[2]; k < hi[2]; k++) {
        for (int j = lo[1]; j < hi[1]; j++) {
            for (int i = lo[0]; i < hi[0]; i++) {
                ptrdiff_t q = grid_index(g, i, j, k);
                rhs[q] += dt * wall[q];
                next[q] = u[q];
            }
        }
    }
    t->compute += MPI_Wtime() - start;
    return cg_solve(op, p, w, -theta * dt, rhs, next, t);
}

static void field_path(char *path, size_t size, const Params *p, const char *name) {
    snprintf(path, size, "%s/%s", p->output, name);
}

static double reduce_max(const Grid *g, double value) {
    double out;
    MPI_Reduce(&value, &out, 1, MPI_DOUBLE, MPI_MAX, 0, g->comm);
    return out;
}

static double reduce_mean(const Grid *g, double value) {
    double out;
    MPI_Reduce(&value, &out, 1, MPI_DOUBLE, MPI_SUM, 0, g->comm);
    return out / g->size;
}

static int run(const Params *params, MPI_Comm world, char *err) {
    Params p = *params;
    Grid g;
    if (grid_create(&g, &p, world, err)) return 1;

    Operator op;
    if (operator_create(&op, &g, &p, err)) {
        grid_free(&g);
        return 1;
    }

    double inv_h2_sum = g.inv_h2[0] + g.inv_h2[1] + g.inv_h2[2];
    double dt_limit = 1.0 / (2.0 * op.alpha_max * inv_h2_sum);
    double dt = p.dt > 0.0 ? p.dt : p.cfl * dt_limit;
    int steps = p.steps;
    if (steps == 0) {
        steps = (int)ceil(p.t_end / dt - 1e-9);
        if (steps < 1) steps = 1;
        dt = p.t_end / steps;
    }
    double t_end = steps * dt;
    double cfl = dt / dt_limit;
    if (p.scheme == SCHEME_EXPLICIT && cfl > 1.0 + 1e-12 && !p.allow_unstable) {
        snprintf(err, ERROR_LEN, "dt %.3e exceeds the explicit stability limit %.3e", dt, dt_limit);
        operator_free(&op);
        grid_free(&g);
        return 1;
    }

    double *u = field_alloc(&g), *next = field_alloc(&g), *rhs = NULL, *wall = NULL;
    if (initialize(&p, &g, u, err)) {
        free(u);
        free(next);
        operator_free(&op);
        grid_free(&g);
        return 1;
    }

    double theta = p.scheme == SCHEME_BACKWARD_EULER ? 1.0 : 0.5;
    CgWorkspace cg = {0};
    if (p.scheme != SCHEME_EXPLICIT) {
        cg_create(&cg, &op, -theta * dt);
        rhs = field_alloc(&g);
        wall = field_alloc(&g);
        double *zero = field_alloc(&g);
        fill_boundary(&g, p.boundary, p.bc_x0, p.bc_x1, zero, false);
        operator_update_all(&op, 1.0, zero, wall);
        free(zero);
    }

    FILE *diag = NULL;
    char path[PATH_LEN + 80];
    if (p.output[0] && g.rank == 0) {
        make_directory(p.output);
        field_path(path, sizeof path, &p, "diag.csv");
        diag = fopen(path, "w");
        if (diag) fprintf(diag, "step,time,energy,min,max,cg_iterations\n");
    }
    MPI_Barrier(g.comm);

    Timers timers = {0};
    Stats initial = field_stats(&g, u);
    if (diag) {
        fprintf(diag, "0,0,%.17g,%.17g,%.17g,0\n", initial.energy, initial.min, initial.max);
    }
    if (p.snapshot_every > 0) {
        field_path(path, sizeof path, &p, "T_00000000.bin");
        double start = MPI_Wtime();
        field_write(&g, path, u);
        timers.io += MPI_Wtime() - start;
    }

    long long cg_total = 0;
    int cg_failures = 0, completed = 0;
    bool diverged = false;
    MPI_Barrier(g.comm);
    double wall_start = MPI_Wtime();
    for (int step = 1; step <= steps; step++) {
        int iterations = 0;
        if (p.scheme == SCHEME_EXPLICIT) {
            step_explicit(&op, &p, dt, u, next, &timers);
        } else {
            iterations = step_implicit(&op, &p, &cg, dt, theta, wall, u, next, rhs, &timers);
            if (iterations < 0) {
                cg_failures++;
                iterations = -iterations - 1;
            }
            cg_total += iterations;
        }
        double *swap = u;
        u = next;
        next = swap;
        completed = step;

        bool due = p.diag_every > 0 && step % p.diag_every == 0;
        if (due || step == steps) {
            Stats s = field_stats(&g, u);
            if (diag) {
                fprintf(diag, "%d,%.17g,%.17g,%.17g,%.17g,%d\n", step, step * dt, s.energy, s.min,
                        s.max, iterations);
            }
            if (!isfinite(s.energy) || !isfinite(s.max) || fabs(s.max) > 1e30) {
                diverged = true;
            }
        }
        if (p.snapshot_every > 0 && !diverged && (step % p.snapshot_every == 0 || step == steps)) {
            char name[64];
            snprintf(name, sizeof name, "T_%08d.bin", step);
            field_path(path, sizeof path, &p, name);
            double start = MPI_Wtime();
            field_write(&g, path, u);
            timers.io += MPI_Wtime() - start;
        }
        if (diverged) break;
    }
    double wall_time = MPI_Wtime() - wall_start;
    wall_time = reduce_max(&g, wall_time);

    Stats final_stats = field_stats(&g, u);
    double l2 = -1.0, linf = -1.0;
    if (p.ic == IC_MODE && !diverged) analytic_error(&p, &g, u, completed * dt, &l2, &linf);
    if (p.output[0] && !diverged) {
        field_path(path, sizeof path, &p, "T_final.bin");
        field_write(&g, path, u);
    }

    double compute_max = reduce_max(&g, timers.compute), compute_mean = reduce_mean(&g, timers.compute);
    double halo_max = reduce_max(&g, timers.halo), halo_mean = reduce_mean(&g, timers.halo);
    double reduce_time = reduce_max(&g, timers.reduce), io_time = reduce_max(&g, timers.io);

    if (g.rank == 0) {
        double updates = (double)g.global_cells * completed;
        double seconds = wall_time - io_time;
        printf("{\"status\":\"%s\",\"ranks\":%d,\"dims\":[%d,%d,%d],\"n\":[%d,%d,%d],",
               diverged ? "diverged" : "ok", g.size, g.dims[0], g.dims[1], g.dims[2], g.global[0],
               g.global[1], g.global[2]);
        printf("\"scheme\":\"%s\",\"halo\":\"%s\",\"boundary\":\"%s\",", SCHEME_NAMES[p.scheme],
               HALO_NAMES[p.halo], BOUNDARY_NAMES[p.boundary]);
        printf("\"steps\":%d,\"completed_steps\":%d,\"dt\":%.17g,\"dt_limit\":%.17g,\"cfl\":%.6g,\"t_end\":%.17g,",
               steps, completed, dt, dt_limit, cfl, t_end);
        printf("\"energy_initial\":%.17g,\"energy_final\":%.17g,\"min\":%.17g,\"max\":%.17g,",
               initial.energy, final_stats.energy, final_stats.min, final_stats.max);
        printf("\"l2_error\":%.17g,\"linf_error\":%.17g,", l2, linf);
        printf("\"cg_iterations\":%lld,\"cg_failures\":%d,", cg_total, cg_failures);
        printf("\"wall\":%.9g,\"compute_max\":%.9g,\"compute_mean\":%.9g,\"halo_max\":%.9g,\"halo_mean\":%.9g,",
               wall_time, compute_max, compute_mean, halo_max, halo_mean);
        printf("\"reduce_max\":%.9g,\"io_max\":%.9g,\"mcups\":%.6g}\n", reduce_time, io_time,
               seconds > 0.0 ? updates / seconds / 1e6 : 0.0);
        fflush(stdout);
    }

    if (diag) fclose(diag);
    if (p.scheme != SCHEME_EXPLICIT) {
        cg_free(&cg);
        free(rhs);
        free(wall);
    }
    free(u);
    free(next);
    operator_free(&op);
    grid_free(&g);
    return 0;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char err[ERROR_LEN] = "";
    Params p;
    int status = params_parse(&p, argc, argv, err);
    if (status == 0) status = run(&p, MPI_COMM_WORLD, err);
    if (status != 0 && rank == 0) {
        fprintf(stderr, "error: %s\n", err);
        fflush(stderr);
    }
    MPI_Finalize();
    return status == 0 ? 0 : 2;
}
