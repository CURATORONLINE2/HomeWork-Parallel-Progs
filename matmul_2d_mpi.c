#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_MPE
#include <mpe.h>
#endif

static void die(int cond, const char* msg, MPI_Comm comm)
{
    if (cond) {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        if (rank == 0) fprintf(stderr, "ERROR: %s\n", msg);
        MPI_Abort(comm, 1);
    }
}

static void* xmalloc(size_t nbytes)
{
    void* p = malloc(nbytes);
    if (!p) {
        fprintf(stderr, "malloc failed (%zu bytes)\n", nbytes);
        exit(1);
    }
    return p;
}

static void fill_matrix(double* M, int rows, int cols)
{
    // Детерминированное заполнение (чтобы можно было сравнивать результаты при желании)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            M[i * cols + j] = (double)((i * 1315423911u + j * 2654435761u) & 1023) / 1024.0;
        }
    }
}

static void local_gemm(double* C, const double* A, const double* B,
                       int m, int k, int n)
{
    // C[m x n] = A[m x k] * B[k x n]
    // Простой O(mkn), для лабы достаточно (BLAS не требуется)
    for (int i = 0; i < m; ++i) {
        double* Ci = C + i * n;
        const double* Ai = A + i * k;
        for (int j = 0; j < n; ++j) Ci[j] = 0.0;

        for (int kk = 0; kk < k; ++kk) {
            const double a = Ai[kk];
            const double* Bk = B + kk * n;
            for (int j = 0; j < n; ++j) {
                Ci[j] += a * Bk[j];
            }
        }
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    MPI_Comm world = MPI_COMM_WORLD;
    int size = 0, rank = 0;
    MPI_Comm_size(world, &size);
    MPI_Comm_rank(world, &rank);

    // Аргументы:
    // mpirun -np P ./a.out n1 n2 n3 [p1 p2]
    // Если p1,p2 не заданы, они выбираются MPI_Dims_create.
    if (argc < 4) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s n1 n2 n3 [p1 p2]\n", argv[0]);
            fprintf(stderr, "Example: mpirun -np 16 %s 4096 4096 4096 4 4\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const int n1 = atoi(argv[1]);
    const int n2 = atoi(argv[2]);
    const int n3 = atoi(argv[3]);

    die(n1 <= 0 || n2 <= 0 || n3 <= 0, "Matrix sizes must be positive.", world);

    int dims[2] = {0, 0};
    int periods[2] = {0, 0};
    int reorder = 1;

    if (argc >= 6) {
        dims[0] = atoi(argv[4]); // p1 (rows of process grid)
        dims[1] = atoi(argv[5]); // p2 (cols of process grid)
        die(dims[0] <= 0 || dims[1] <= 0, "p1 and p2 must be positive.", world);
        die(dims[0] * dims[1] != size, "p1*p2 must equal number of MPI processes.", world);
    } else {
        MPI_Dims_create(size, 2, dims);
    }

    const int p1 = dims[0];
    const int p2 = dims[1];

    die((p1 * p2) != size, "Internal error: dims product != size.", world);
    die((n1 % p1) != 0, "Requirement: n1 must be divisible by p1.", world);
    die((n3 % p2) != 0, "Requirement: n3 must be divisible by p2.", world);

    MPI_Comm comm2d = MPI_COMM_NULL;
    MPI_Cart_create(world, 2, dims, periods, reorder, &comm2d);
    die(comm2d == MPI_COMM_NULL, "MPI_Cart_create failed.", world);

    int rank2d = 0;
    MPI_Comm_rank(comm2d, &rank2d);

    int coords[2] = {0, 0};
    MPI_Cart_coords(comm2d, rank2d, 2, coords);
    const int my_i = coords[0]; // along p1
    const int my_j = coords[1]; // along p2

    // Подкоммуникаторы строк/столбцов для broadcast
    MPI_Comm row_comm = MPI_COMM_NULL;
    MPI_Comm col_comm = MPI_COMM_NULL;

    // row_comm: фиксируем i, меняем j
    MPI_Comm_split(comm2d, my_i, my_j, &row_comm);
    // col_comm: фиксируем j, меняем i
    MPI_Comm_split(comm2d, my_j, my_i, &col_comm);

    die(row_comm == MPI_COMM_NULL || col_comm == MPI_COMM_NULL, "MPI_Comm_split failed.", comm2d);

    // Локальные размеры
    const int a_rows = n1 / p1;   // строк в полосе A
    const int a_cols = n2;        // ширина A
    const int b_rows = n2;        // высота B
    const int b_cols = n3 / p2;   // столбцов в полосе B
    const int c_rows = a_rows;
    const int c_cols = b_cols;

    // Буферы
    double* A_full = NULL;
    double* B_full = NULL;
    double* C_full = NULL;

    // Полосы/блоки
    double* A_strip = (double*)xmalloc((size_t)a_rows * (size_t)a_cols * sizeof(double));
    double* B_strip = (double*)xmalloc((size_t)b_rows * (size_t)b_cols * sizeof(double));
    double* C_block = (double*)xmalloc((size_t)c_rows * (size_t)c_cols * sizeof(double));

    if (rank2d == 0) {
        A_full = (double*)xmalloc((size_t)n1 * (size_t)n2 * sizeof(double));
        B_full = (double*)xmalloc((size_t)n2 * (size_t)n3 * sizeof(double));
        C_full = (double*)xmalloc((size_t)n1 * (size_t)n3 * sizeof(double));
        fill_matrix(A_full, n1, n2);
        fill_matrix(B_full, n2, n3);
    }

#ifdef USE_MPE
    MPE_Init_log();
    int ev_scatterA_s, ev_scatterA_e, ev_scatterB_s, ev_scatterB_e;
    int ev_bcastA_s, ev_bcastA_e, ev_bcastB_s, ev_bcastB_e;
    int ev_compute_s, ev_compute_e, ev_gather_s, ev_gather_e;
    MPE_Log_get_state_eventIDs(&ev_scatterA_s, &ev_scatterA_e);
    MPE_Log_get_state_eventIDs(&ev_scatterB_s, &ev_scatterB_e);
    MPE_Log_get_state_eventIDs(&ev_bcastA_s, &ev_bcastA_e);
    MPE_Log_get_state_eventIDs(&ev_bcastB_s, &ev_bcastB_e);
    MPE_Log_get_state_eventIDs(&ev_compute_s,  &ev_compute_e);
    MPE_Log_get_state_eventIDs(&ev_gather_s,   &ev_gather_e);
    MPE_Describe_state(ev_scatterA_s, ev_scatterA_e, "scatter_A", "red");
    MPE_Describe_state(ev_scatterB_s, ev_scatterB_e, "scatter_B", "blue");
    MPE_Describe_state(ev_bcastA_s,   ev_bcastA_e,   "bcast_A",   "green");
    MPE_Describe_state(ev_bcastB_s,   ev_bcastB_e,   "bcast_B",   "yellow");
    MPE_Describe_state(ev_compute_s,  ev_compute_e,  "compute",   "white");
    MPE_Describe_state(ev_gather_s,   ev_gather_e,   "gather_C",  "cyan");
#endif

    MPI_Barrier(comm2d);
    const double t0 = MPI_Wtime();

    // 1) Scatter A: по первому столбцу (j==0) раздаем полосы A
    // Для этого создаем коммуникатор col0 (процессы с j==0).
    MPI_Comm col0_comm = MPI_COMM_NULL;
    MPI_Comm_split(comm2d, (my_j == 0) ? 1 : MPI_UNDEFINED, my_i, &col0_comm);

#ifdef USE_MPE
    if (my_j == 0) MPE_Log_event(ev_scatterA_s, 0, NULL);
#endif

    if (my_j == 0) {
        // В col0_comm ранги упорядочены по my_i (key=my_i)
        int col0_rank = 0;
        MPI_Comm_rank(col0_comm, &col0_rank);

        // Root для scatter A должен быть тот процесс, который является (0,0).
        // В col0_comm это тот, у кого my_i=0 => col0_rank=0.
        MPI_Scatter(
            A_full, a_rows * a_cols, MPI_DOUBLE,
            A_strip, a_rows * a_cols, MPI_DOUBLE,
            0, col0_comm
        );
    }

#ifdef USE_MPE
    if (my_j == 0) MPE_Log_event(ev_scatterA_e, 0, NULL);
#endif

    // 2) Scatter B: по первой строке (i==0) раздаем вертикальные полосы B
    // Тут важно: полосы B в памяти root не непрерывны, нужен derived datatype.
    MPI_Comm row0_comm = MPI_COMM_NULL;
    MPI_Comm_split(comm2d, (my_i == 0) ? 1 : MPI_UNDEFINED, my_j, &row0_comm);

#ifdef USE_MPE
    if (my_i == 0) MPE_Log_event(ev_scatterB_s, 0, NULL);
#endif

    if (my_i == 0) {
        int row0_rank = 0;
        MPI_Comm_rank(row0_comm, &row0_rank);

        MPI_Datatype col_type = MPI_DATATYPE_NULL;
        // Описываем вертикальную полосу шириной b_cols в матрице B_full (n2 x n3)
        // как n2 блоков по b_cols подряд, с шагом n3 между началом строк.
        MPI_Type_vector(n2, b_cols, n3, MPI_DOUBLE, &col_type);

        MPI_Datatype col_type_resized = MPI_DATATYPE_NULL;
        // Чтобы Scatter выбирал следующую полосу, делаем extent = b_cols*sizeof(double)
        MPI_Type_create_resized(col_type, 0, (MPI_Aint)(b_cols * (int)sizeof(double)), &col_type_resized);
        MPI_Type_commit(&col_type_resized);

        // sendbuf: B_full, sendtype: col_type_resized, sendcount = 1 на процесс
        // recvbuf: непрерывный буфер B_strip (n2 x b_cols)
        MPI_Scatter(
            B_full, 1, col_type_resized,
            B_strip, n2 * b_cols, MPI_DOUBLE,
            0, row0_comm
        );

        MPI_Type_free(&col_type_resized);
        MPI_Type_free(&col_type);
    }

#ifdef USE_MPE
    if (my_i == 0) MPE_Log_event(ev_scatterB_e, 0, NULL);
#endif

    // 3) Broadcast A вдоль строк (от j==0 ко всем j)
#ifdef USE_MPE
    MPE_Log_event(ev_bcastA_s, 0, NULL);
#endif
    // В row_comm root = процесс с my_j==0, в row_comm он имеет rank 0 (key=my_j).
    MPI_Bcast(A_strip, a_rows * a_cols, MPI_DOUBLE, 0, row_comm);
#ifdef USE_MPE
    MPE_Log_event(ev_bcastA_e, 0, NULL);
#endif

    // 4) Broadcast B вдоль столбцов (от i==0 ко всем i)
#ifdef USE_MPE
    MPE_Log_event(ev_bcastB_s, 0, NULL);
#endif
    // В col_comm root = процесс с my_i==0, в col_comm он имеет rank 0 (key=my_i).
    MPI_Bcast(B_strip, b_rows * b_cols, MPI_DOUBLE, 0, col_comm);
#ifdef USE_MPE
    MPE_Log_event(ev_bcastB_e, 0, NULL);
#endif

    // 5) Локальное умножение блока
    const double t_comp0 = MPI_Wtime();
#ifdef USE_MPE
    MPE_Log_event(ev_compute_s, 0, NULL);
#endif
    local_gemm(C_block, A_strip, B_strip, c_rows, n2, c_cols);
#ifdef USE_MPE
    MPE_Log_event(ev_compute_e, 0, NULL);
#endif
    const double t_comp1 = MPI_Wtime();
    const double comp_time = t_comp1 - t_comp0;

    // 6) Сбор C на (0,0): блоки C_block -> C_full
#ifdef USE_MPE
    MPE_Log_event(ev_gather_s, 0, NULL);
#endif

    if (rank2d == 0) {
        // Root принимает блоки в правильные места.
        // Опишем тип "блок c_rows x c_cols" внутри большой C_full (n1 x n3).
        MPI_Datatype block = MPI_DATATYPE_NULL;
        MPI_Type_vector(c_rows, c_cols, n3, MPI_DOUBLE, &block);

        MPI_Datatype block_resized = MPI_DATATYPE_NULL;
        // extent по горизонтали = c_cols*sizeof(double) (чтобы блоки в строке решетки шли подряд)
        MPI_Type_create_resized(block, 0, (MPI_Aint)(c_cols * (int)sizeof(double)), &block_resized);
        MPI_Type_commit(&block_resized);

        // Сдвиги (displs) в единицах block_resized
        int* recvcounts = (int*)xmalloc((size_t)size * sizeof(int));
        int* displs     = (int*)xmalloc((size_t)size * sizeof(int));

        for (int r = 0; r < size; ++r) {
            int cc[2];
            MPI_Cart_coords(comm2d, r, 2, cc);
            int ii = cc[0];
            int jj = cc[1];
            // Блок (ii,jj) начинается с строки ii*c_rows и столбца jj*c_cols.
            // В единицах resized-типа смещение:
            // displ = ii*(c_rows*n3/c_cols) + jj
            // так как один "extent" = c_cols элементов
            displs[r] = ii * (c_rows * n3 / c_cols) + jj;
            recvcounts[r] = 1;
        }

        // Root тоже отправляет свой C_block
        MPI_Gatherv(
            C_block, c_rows * c_cols, MPI_DOUBLE,
            C_full, recvcounts, displs, block_resized,
            0, comm2d
        );

        free(recvcounts);
        free(displs);
        MPI_Type_free(&block_resized);
        MPI_Type_free(&block);
    } else {
        MPI_Gatherv(
            C_block, c_rows * c_cols, MPI_DOUBLE,
            NULL, NULL, NULL, MPI_DATATYPE_NULL,
            0, comm2d
        );
    }

#ifdef USE_MPE
    MPE_Log_event(ev_gather_e, 0, NULL);
#endif

    MPI_Barrier(comm2d);
    const double t1 = MPI_Wtime();
    const double total_time = t1 - t0;

    // Сведем времена:
    double total_time_max = 0.0;
    double comp_time_max  = 0.0;
    MPI_Reduce(&total_time, &total_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm2d);
    MPI_Reduce(&comp_time,  &comp_time_max,  1, MPI_DOUBLE, MPI_MAX, 0, comm2d);

    // Для расчета ускорения/эффективности удобно: запуск на 1 процессе запомнить отдельно.
    // Но программа сама не хранит baseline, поэтому в отчетах обычно берут T1 из отдельного прогона.
    if (rank2d == 0) {
        printf("np=%d\ngrid=%dx%d\nn1=%d n2=%d n3=%d\nT_total_max=%.6f s \nT_comp_max=%.6f s\n",
               size, p1, p2, n1, n2, n3, total_time_max, comp_time_max);
        fflush(stdout);
    }

#ifdef USE_MPE
    // Файл логов появится после финализации; обычно: clog2slog2 и jumpshot.
    MPE_Finish_log("mpe_matmul2d");
#endif

    if (col0_comm != MPI_COMM_NULL) MPI_Comm_free(&col0_comm);
    if (row0_comm != MPI_COMM_NULL) MPI_Comm_free(&row0_comm);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&comm2d);

    free(A_strip);
    free(B_strip);
    free(C_block);
    if (rank2d == 0) {
        free(A_full);
        free(B_full);
        free(C_full);
    }

    MPI_Finalize();
    return 0;
}

