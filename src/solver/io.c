#include "heat.h"

#include <stdio.h>

int field_read(const Grid *g, const char *path, double *f, char *err) {
    MPI_File file;
    if (MPI_File_open(g->comm, path, MPI_MODE_RDONLY, MPI_INFO_NULL, &file) != MPI_SUCCESS) {
        snprintf(err, ERROR_LEN, "could not open %s", path);
        return 1;
    }
    MPI_Offset size;
    MPI_File_get_size(file, &size);
    MPI_Offset expected = (MPI_Offset)g->global_cells * (MPI_Offset)sizeof(double);
    if (size != expected) {
        snprintf(err, ERROR_LEN, "%s holds %lld bytes, expected %lld", path, (long long)size,
                 (long long)expected);
        MPI_File_close(&file);
        return 1;
    }
    char native[] = "native";
    MPI_File_set_view(file, 0, MPI_DOUBLE, g->interior_file, native, MPI_INFO_NULL);
    int status = MPI_File_read_all(file, f, 1, g->interior_mem, MPI_STATUS_IGNORE);
    MPI_File_close(&file);
    if (status != MPI_SUCCESS) {
        snprintf(err, ERROR_LEN, "could not read %s", path);
        return 1;
    }
    return 0;
}

int field_write(const Grid *g, const char *path, const double *f) {
    MPI_File file;
    int mode = MPI_MODE_CREATE | MPI_MODE_WRONLY;
    if (MPI_File_open(g->comm, path, mode, MPI_INFO_NULL, &file) != MPI_SUCCESS) return 1;
    MPI_File_set_size(file, 0);
    char native[] = "native";
    MPI_File_set_view(file, 0, MPI_DOUBLE, g->interior_file, native, MPI_INFO_NULL);
    int status = MPI_File_write_all(file, f, 1, g->interior_mem, MPI_STATUS_IGNORE);
    MPI_File_close(&file);
    return status != MPI_SUCCESS;
}
