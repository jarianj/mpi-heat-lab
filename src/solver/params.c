#include "heat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *name;
    int value;
} Choice;

static const Choice SCHEMES[] = {{"explicit", SCHEME_EXPLICIT},
                                 {"cn", SCHEME_CRANK_NICOLSON},
                                 {"be", SCHEME_BACKWARD_EULER},
                                 {NULL, 0}};
static const Choice HALOS[] = {{"chain", HALO_CHAIN},
                               {"sendrecv", HALO_SENDRECV},
                               {"overlap", HALO_OVERLAP},
                               {NULL, 0}};
static const Choice BOUNDARIES[] = {{"neumann", BC_NEUMANN},
                                    {"dirichlet", BC_DIRICHLET},
                                    {"periodic", BC_PERIODIC},
                                    {NULL, 0}};
static const Choice ICS[] = {{"uniform", IC_UNIFORM},
                             {"gaussian", IC_GAUSSIAN},
                             {"step", IC_STEP},
                             {"mode", IC_MODE},
                             {"file", IC_FILE},
                             {NULL, 0}};
static const Choice MEDIA[] = {{"uniform", MEDIUM_UNIFORM},
                               {"layers", MEDIUM_LAYERS},
                               {"inclusion", MEDIUM_INCLUSION},
                               {"file", MEDIUM_FILE},
                               {NULL, 0}};

static int pick(const Choice *c, const char *key, const char *value, int *out, char *err) {
    for (; c->name; c++) {
        if (strcmp(c->name, value) == 0) {
            *out = c->value;
            return 0;
        }
    }
    snprintf(err, ERROR_LEN, "invalid value '%s' for %s", value, key);
    return 1;
}

static void defaults(Params *p) {
    memset(p, 0, sizeof *p);
    p->n[0] = 256;
    p->n[1] = p->n[2] = 1;
    p->scheme = SCHEME_EXPLICIT;
    p->halo = HALO_OVERLAP;
    p->boundary = BC_NEUMANN;
    p->ic = IC_GAUSSIAN;
    p->medium = MEDIUM_UNIFORM;
    p->alpha = p->alpha2 = 1.0;
    p->t_end = 0.01;
    p->cfl = 0.9;
    p->cg_tol = 1e-10;
    p->cg_max = 10000;
    p->diag_every = 0;
}

int params_parse(Params *p, int argc, char **argv, char *err) {
    defaults(p);
    for (int a = 1; a < argc; a += 2) {
        const char *key = argv[a];
        if (strcmp(key, "--allow-unstable") == 0) {
            p->allow_unstable = true;
            a--;
            continue;
        }
        if (a + 1 >= argc) {
            snprintf(err, ERROR_LEN, "missing value for %s", key);
            return 1;
        }
        const char *value = argv[a + 1];
        int choice = 0;
        if (strcmp(key, "--nx") == 0) p->n[0] = atoi(value);
        else if (strcmp(key, "--ny") == 0) p->n[1] = atoi(value);
        else if (strcmp(key, "--nz") == 0) p->n[2] = atoi(value);
        else if (strcmp(key, "--kx") == 0) p->mode[0] = atoi(value);
        else if (strcmp(key, "--ky") == 0) p->mode[1] = atoi(value);
        else if (strcmp(key, "--kz") == 0) p->mode[2] = atoi(value);
        else if (strcmp(key, "--alpha") == 0) p->alpha = atof(value);
        else if (strcmp(key, "--alpha2") == 0) p->alpha2 = atof(value);
        else if (strcmp(key, "--bc-x0") == 0) p->bc_x0 = atof(value);
        else if (strcmp(key, "--bc-x1") == 0) p->bc_x1 = atof(value);
        else if (strcmp(key, "--t-end") == 0) p->t_end = atof(value);
        else if (strcmp(key, "--dt") == 0) p->dt = atof(value);
        else if (strcmp(key, "--cfl") == 0) p->cfl = atof(value);
        else if (strcmp(key, "--cg-tol") == 0) p->cg_tol = atof(value);
        else if (strcmp(key, "--cg-max") == 0) p->cg_max = atoi(value);
        else if (strcmp(key, "--steps") == 0) p->steps = atoi(value);
        else if (strcmp(key, "--diag-every") == 0) p->diag_every = atoi(value);
        else if (strcmp(key, "--snapshot-every") == 0) p->snapshot_every = atoi(value);
        else if (strcmp(key, "--ic-file") == 0) snprintf(p->ic_file, PATH_LEN, "%s", value);
        else if (strcmp(key, "--alpha-file") == 0) snprintf(p->alpha_file, PATH_LEN, "%s", value);
        else if (strcmp(key, "--output") == 0) snprintf(p->output, PATH_LEN, "%s", value);
        else if (strcmp(key, "--dims") == 0) {
            if (sscanf(value, "%d,%d,%d", &p->dims[0], &p->dims[1], &p->dims[2]) != 3) {
                snprintf(err, ERROR_LEN, "--dims expects a,b,c");
                return 1;
            }
        } else if (strcmp(key, "--scheme") == 0) {
            if (pick(SCHEMES, key, value, &choice, err)) return 1;
            p->scheme = (Scheme)choice;
        } else if (strcmp(key, "--halo") == 0) {
            if (pick(HALOS, key, value, &choice, err)) return 1;
            p->halo = (HaloMode)choice;
        } else if (strcmp(key, "--bc") == 0) {
            if (pick(BOUNDARIES, key, value, &choice, err)) return 1;
            p->boundary = (Boundary)choice;
        } else if (strcmp(key, "--ic") == 0) {
            if (pick(ICS, key, value, &choice, err)) return 1;
            p->ic = (InitialCondition)choice;
        } else if (strcmp(key, "--medium") == 0) {
            if (pick(MEDIA, key, value, &choice, err)) return 1;
            p->medium = (Medium)choice;
        } else {
            snprintf(err, ERROR_LEN, "unknown option %s", key);
            return 1;
        }
    }

    for (int a = 0; a < AXES; a++) {
        if (p->n[a] < 1) {
            snprintf(err, ERROR_LEN, "grid sizes must be positive");
            return 1;
        }
    }
    if (p->alpha <= 0 || p->alpha2 <= 0) {
        snprintf(err, ERROR_LEN, "diffusivity must be positive");
        return 1;
    }
    if (p->cfl <= 0 || p->dt < 0 || p->t_end <= 0 || p->steps < 0) {
        snprintf(err, ERROR_LEN, "cfl, dt, t-end and steps must be positive");
        return 1;
    }
    if (p->halo == HALO_CHAIN && p->boundary == BC_PERIODIC) {
        snprintf(err, ERROR_LEN, "chain halo exchange deadlocks on a ring; use sendrecv or overlap");
        return 1;
    }
    if (p->ic == IC_FILE && !p->ic_file[0]) {
        snprintf(err, ERROR_LEN, "--ic file requires --ic-file");
        return 1;
    }
    if (p->medium == MEDIUM_FILE && !p->alpha_file[0]) {
        snprintf(err, ERROR_LEN, "--medium file requires --alpha-file");
        return 1;
    }
    if (p->snapshot_every > 0 && !p->output[0]) {
        snprintf(err, ERROR_LEN, "--snapshot-every requires --output");
        return 1;
    }
    return 0;
}
