// softmax_cuda.cu
// Compilar: nvcc -O3 -std=c++14 softmax_cuda.cu -o softmax_cuda
// Uso:      ./softmax_cuda [--precision fp32|fp16] [--n N] [--reps R] [--measure kernel|end-to-end]
//           (por defecto: fp32, 100, 10000, kernel)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Las llamadas a CUDA no lanzan excepciones: devuelven un código de error que
// hay que revisar. La macro aborta mostrando archivo y línea del fallo.
// El do { } while (0) permite usarla como una sola sentencia (p. ej. tras un if).
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err));                             \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// THREADS debe ser potencia de 2: la reducción en árbol divide el número de
// hilos activos entre 2 en cada nivel (256 -> 128 -> ... -> 1).
// 256 hilos = 8 warps de 32, suficiente para n <= 1024 (máx. 4 elementos/hilo).
#define THREADS 256
// Tamaño máximo del vector; fija el tamaño estático de 'ex' en memoria compartida.
#define MAX_N 1024

// ---------------- Conversiones en el dispositivo ----------------
// Sobrecargas para que el kernel con plantilla lea/escriba T sin saber si es
// float o __half: siempre trabaja en float y solo convierte en los bordes.
// __forceinline__ evita el costo de una llamada a función en el kernel.
__device__ __forceinline__ float load_f(float v) { return v; }
__device__ __forceinline__ float load_f(__half v) { return __half2float(v); }
__device__ __forceinline__ void store_f(float *p, float v) { *p = v; }
__device__ __forceinline__ void store_f(__half *p, float v) { *p = __float2half(v); }

// ---------------- Kernel: un solo vector, un solo bloque ----------------
// x e y en T; maximo, exp y suma en float; exponenciales guardadas en float.
// Se lanza con <<<1, THREADS>>>: un solo bloque de 256 hilos.
// Un solo bloque porque el máximo y la suma son reducciones sobre TODO el
// vector, y __syncthreads() y la memoria compartida solo funcionan dentro de un
// bloque. Con varios bloques habría que combinar resultados parciales con otro
// kernel o con atómicas. Con n <= 1024 un bloque basta y es la versión de una
// fila del kernel del profesor.
template <typename T>
__global__ void softmax_kernel(const T *input, T *output, int n) {
    // Memoria compartida: la ven todos los hilos del bloque y es mucho más
    // rápida que la global.
    // cache: un valor parcial por hilo (máximo y luego suma) para la reducción.
    __shared__ float cache[THREADS];
    // ex: exponenciales en float (4 KB). Se guardan aquí para no recalcular
    // expf en el paso 3 y para no redondearlas a FP16 antes de normalizar.
    __shared__ float ex[MAX_N];

    int tid = threadIdx.x;

    // Paso 1: maximo con reduccion
    // Bucle con paso blockDim.x (grid-stride dentro del bloque): el hilo tid
    // procesa los elementos tid, tid+256, tid+512, tid+768. Así 256 hilos
    // cubren hasta 1024 elementos, y hilos consecutivos leen direcciones
    // consecutivas (accesos coalescentes a memoria global).
    // -INFINITY es el neutro del máximo: un hilo sin elementos (tid >= n) no
    // altera el resultado.
    // Restar luego el máximo da la forma estable: exponentes <= 0, sin overflow.
    float local_max = -INFINITY;
    for (int col = tid; col < n; col += blockDim.x) {
        local_max = fmaxf(local_max, load_f(input[col]));
    }

    cache[tid] = local_max;
    // Todos los máximos parciales deben estar escritos en cache antes de que
    // la reducción empiece a leer los de otros hilos.
    __syncthreads();

    // Reducción en árbol: en cada nivel la mitad inferior de hilos combina su
    // valor con el de su pareja a distancia 'stride'. Toma log2(256) = 8 niveles
    // en vez de 255 pasos secuenciales. Al final, cache[0] tiene el máximo.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            cache[tid] = fmaxf(cache[tid], cache[tid + stride]);
        }
        // Cada nivel debe terminar antes del siguiente, que lee lo escrito en
        // este. Va fuera del if: todos los hilos del bloque deben alcanzarlo.
        __syncthreads();
    }

    float row_max = cache[0];
    // Todos deben haber leído cache[0] antes de que algún hilo lo sobrescriba
    // con su suma parcial abajo; sin esta barrera un hilo rápido podría
    // reemplazar el máximo antes de que otro hilo lo lea.
    __syncthreads();  // evita la carrera al reutilizar cache para la suma

    // Paso 2: exp(x_i - max) y reduccion de la suma
    // Mismo recorrido con paso blockDim.x; 0 es el neutro de la suma.
    float local_sum = 0.0f;
    for (int col = tid; col < n; col += blockDim.x) {
        float e = expf(load_f(input[col]) - row_max);
        ex[col] = e;
        local_sum += e;
    }

    cache[tid] = local_sum;
    // Todas las sumas parciales escritas antes de empezar la reducción.
    __syncthreads();

    // Misma reducción en árbol, ahora sumando. Al final cache[0] es la suma total.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        // Mismo motivo que en el máximo: separa los niveles del árbol. La
        // barrera del último nivel también garantiza que cache[0] es final.
        __syncthreads();
    }

    float row_sum = cache[0];

    // Paso 3: normalizacion
    // No hace falta otra barrera: cada hilo lee solo las posiciones de 'ex' que
    // él mismo escribió (mismo patrón col = tid + k*blockDim.x) y cache ya no
    // se modifica. Solo aquí se convierte a T (único redondeo de la salida).
    for (int col = tid; col < n; col += blockDim.x) {
        store_f(&output[col], ex[col] / row_sum);
    }
}

// ---------------- Conversiones en el host ----------------
// __float2half/__half2float existen también en el host en CUDA 10.2. Las
// sobrecargas dejan que run<T> convierta sin distinguir el tipo.
static void from_double(double d, float &o) { o = static_cast<float>(d); }
static void from_double(double d, __half &o) { o = __float2half(static_cast<float>(d)); }
static double to_double(float v) { return static_cast<double>(v); }
static double to_double(__half v) { return static_cast<double>(__half2float(v)); }

// Referencia en double con std::exp
// FP64 tiene un error despreciable frente a FP32/FP16; se usa como valor
// verdadero para medir el error de la GPU.
static void softmax_ref(const std::vector<double> &x, std::vector<double> &y) {
    const int n = static_cast<int>(x.size());
    double m = x[0];
    for (int i = 1; i < n; ++i) m = std::max(m, x[i]);
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        y[i] = std::exp(x[i] - m);
        s += y[i];
    }
    for (int i = 0; i < n; ++i) y[i] /= s;
}

template <typename T>
static int run(const char *prec, int n, int R, const std::string &measure) {
    // Datos: siempre el mismo vector en double
    // Misma semilla y distribución que en la CPU: el vector es idéntico y los
    // resultados de ambas plataformas son comparables.
    std::mt19937 gen(22);
    std::uniform_real_distribution<double> dist(-4.0, 4.0);
    std::vector<double> xd(n);
    for (int i = 0; i < n; ++i) xd[i] = dist(gen);

    // Referencia con el vector original en double (antes de convertir)
    // Así el error medido incluye el redondeo de la entrada a T.
    std::vector<double> ref(n);
    softmax_ref(xd, ref);

    // Conversion a T en el host
    std::vector<T> h_x(n), h_y(n);
    for (int i = 0; i < n; ++i) from_double(xd[i], h_x[i]);

    // Inicializacion del contexto CUDA
    // El contexto se crea de forma perezosa en la primera llamada a CUDA y
    // puede tardar cientos de ms; forzarlo aquí evita que ese costo caiga
    // dentro de la medición.
    CUDA_CHECK(cudaFree(0));

    size_t bytes = static_cast<size_t>(n) * sizeof(T);
    T *d_x = nullptr;
    T *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    // Salida en cero: si el kernel no escribiera, la verificación lo detectaría.
    CUDA_CHECK(cudaMemset(d_y, 0, bytes));

    // ---------------- Verificacion ----------------
    // Configuración de lanzamiento: 1 bloque, THREADS (256) hilos.
    softmax_kernel<T><<<1, THREADS>>>(d_x, d_y, n);
    // El lanzamiento no devuelve error directamente: cudaGetLastError detecta
    // fallos de configuración, y la sincronización, errores durante la ejecución.
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));

    // Se verifica en double en el host; vmax/rmax inician en -1 porque toda
    // salida de softmax es >= 0, y ref[i] > 0 evita dividir entre cero.
    double suma = 0.0, err_abs = 0.0, err_rel = 0.0;
    bool rango_ok = true;
    int am = 0, am_ref = 0;
    double vmax = -1.0, rmax = -1.0;
    for (int i = 0; i < n; ++i) {
        double v = to_double(h_y[i]);
        suma += v;
        if (v < 0.0 || v > 1.0) rango_ok = false;
        if (v > vmax) { vmax = v; am = i; }
        if (ref[i] > rmax) { rmax = ref[i]; am_ref = i; }
        double ea = std::fabs(v - ref[i]);
        double er = ea / ref[i];
        if (ea > err_abs) err_abs = ea;
        if (er > err_rel) err_rel = er;
    }

    std::printf("[%s] n=%d suma=%.6f rango=%s argmax=%s err_abs=%.1e err_rel=%.1e\n",
                prec, n, suma, rango_ok ? "OK" : "FAIL",
                am == am_ref ? "OK" : "FAIL", err_abs, err_rel);

    // ---------------- Medicion ----------------
    // Protocolo: calentamiento descartado + 5 lotes de R llamadas; se guarda el
    // promedio por llamada de cada lote.
    const int M = 5;
    std::vector<double> t(M);

    if (measure == "kernel") {
        // Modo kernel: solo el cómputo en la GPU, con los datos ya en el
        // dispositivo. Los eventos CUDA se registran en el propio stream de la
        // GPU, así que miden el tiempo del dispositivo y no el del host.
        cudaEvent_t ev0, ev1;
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));

        // Calentamiento (descartado)
        // Carga el código del kernel y deja la GPU en su frecuencia de trabajo.
        for (int r = 0; r < R; ++r) {
            softmax_kernel<T><<<1, THREADS>>>(d_x, d_y, n);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int k = 0; k < M; ++k) {
            // Los lanzamientos son asíncronos y se encolan en orden: ev0 queda
            // antes del primero y ev1 después del último del lote.
            CUDA_CHECK(cudaEventRecord(ev0));
            for (int r = 0; r < R; ++r) {
                softmax_kernel<T><<<1, THREADS>>>(d_x, d_y, n);
                CUDA_CHECK(cudaGetLastError());
            }
            CUDA_CHECK(cudaEventRecord(ev1));
            // Espera a que la GPU llegue a ev1, es decir, que termine el lote.
            CUDA_CHECK(cudaEventSynchronize(ev1));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
            // ms -> us, dividido entre R para obtener el tiempo por llamada.
            t[k] = static_cast<double>(ms) * 1000.0 / R;
            std::printf("medicion %d: %.4f us/llamada\n", k + 1, t[k]);
        }

        CUDA_CHECK(cudaEventDestroy(ev0));
        CUDA_CHECK(cudaEventDestroy(ev1));
    } else {
        // Modo end-to-end: costo completo visto por la aplicación (copia de
        // entrada, kernel, sincronización y copia de salida). Como incluye
        // trabajo del host, se mide con el reloj del host.
        // Calentamiento (descartado)
        for (int r = 0; r < R; ++r) {
            CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
            softmax_kernel<T><<<1, THREADS>>>(d_x, d_y, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
        }

        for (int k = 0; k < M; ++k) {
            // steady_clock es monótono; cada iteración termina sincronizada,
            // así que t1 se toma con todo el trabajo de la GPU ya completado.
            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < R; ++r) {
                CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
                softmax_kernel<T><<<1, THREADS>>>(d_x, d_y, n);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
            }
            auto t1 = std::chrono::steady_clock::now();
            t[k] = std::chrono::duration<double, std::micro>(t1 - t0).count() / R;
            std::printf("medicion %d: %.4f us/llamada\n", k + 1, t[k]);
        }
    }

    // Copia ordenada para min/mediana/max sin alterar el orden de t en el CSV.
    std::vector<double> ts = t;
    std::sort(ts.begin(), ts.end());
    std::printf("min=%.4f us  mediana=%.4f us  max=%.4f us  (R=%d, %s)\n",
                ts[0], ts[M / 2], ts[M - 1], R, measure.c_str());

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    // ---------------- CSV ----------------
    // Encabezado solo si el archivo no existe; modo append para acumular corridas.
    FILE *chk = std::fopen("gpu_results.csv", "r");
    bool existe = (chk != nullptr);
    if (chk) std::fclose(chk);

    FILE *f = std::fopen("gpu_results.csv", "a");
    if (!f) {
        std::fprintf(stderr, "No se pudo abrir gpu_results.csv\n");
        return 1;
    }
    if (!existe) std::fprintf(f, "precision,n,reps,measure,us_por_llamada,err_abs,err_rel\n");
    for (int k = 0; k < M; ++k) {
        std::fprintf(f, "%s,%d,%d,%s,%.6f,%.6e,%.6e\n", prec, n, R, measure.c_str(),
                     t[k], err_abs, err_rel);
    }
    std::fclose(f);

    return 0;
}

int main(int argc, char **argv) {
    // Valores por defecto si no se pasan banderas.
    std::string prec = "fp32";
    int n = 100;
    int R = 10000;
    std::string measure = "kernel";

    // Cada bandera consume su valor (++i); una bandera desconocida o sin valor
    // invalida n para que se imprima el uso.
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--precision") == 0 && i + 1 < argc) {
            prec = argv[++i];
        } else if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            n = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            R = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--measure") == 0 && i + 1 < argc) {
            measure = argv[++i];
        } else {
            n = -1;  // argumento invalido
            break;
        }
    }

    // n > MAX_N desbordaría 'ex' en memoria compartida, por eso se rechaza.
    if ((prec != "fp32" && prec != "fp16") || n < 1 || n > MAX_N || R < 1 ||
        (measure != "kernel" && measure != "end-to-end")) {
        std::fprintf(stderr,
                     "Uso: %s [--precision fp32|fp16] [--n N<=1024] [--reps R] "
                     "[--measure kernel|end-to-end]\n",
                     argv[0]);
        return 1;
    }

    // Cada rama instancia el kernel y run<> con un tipo distinto en compilación.
    if (prec == "fp16") return run<__half>("fp16", n, R, measure);
    return run<float>("fp32", n, R, measure);
}
