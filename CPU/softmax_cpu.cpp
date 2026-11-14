// softmax_cpu.cpp — softmax de un vector en la CPU de la Jetson Nano (Cortex-A57).
// Compilar: g++ -O3 -std=c++17 -march=native softmax_cpu.cpp -o softmax_cpu -lm
// Uso:      ./softmax_cpu --precision fp32|fp16 --n N(<=1024) --reps R
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

// __fp16 es el tipo de media precisión de GCC en ARM. Es solo un formato de
// almacenamiento: cualquier operación con él se promueve a float, y no existe
// expf para FP16. Por eso FP16 aquí reduce memoria, pero el cálculo es en float.
typedef __fp16 half_t;

// Softmax de un vector de tamano n (mismos 3 pasos que el kernel del profesor).
// x e y se guardan en T; maximo, exponenciales y suma se calculan en float.
// 'e' es un buffer float externo: guardar las exponenciales en float (y no en T)
// evita redondearlas a 16 bits antes de dividir, y reservarlo fuera evita
// asignar memoria en cada llamada (lo cual contaminaría la medición).
template <typename T>
void softmax(const T *x, T *y, float *e, int n) {
    // Paso 1: maximo
    // Forma estable: softmax(x) = softmax(x - max), porque el factor e^(-max)
    // se cancela entre numerador y denominador. Restar el máximo deja todos los
    // exponentes <= 0, así exp() queda en (0, 1] y nunca desborda.
    float m = static_cast<float>(x[0]);
    for (int i = 1; i < n; ++i) {
        m = fmaxf(m, static_cast<float>(x[i]));
    }

    // Paso 2: exp(x_i - max) y suma
    // Al menos un término vale exp(0) = 1, así que la suma es >= 1 y la
    // división del paso 3 nunca es entre cero ni entre un número diminuto.
    float s = 0.0f;
    for (int i = 0; i < n; ++i) {
        float v = expf(static_cast<float>(x[i]) - m);
        e[i] = v;
        s += v;
    }

    // Paso 3: normalizacion
    // Solo aquí se convierte a T: en FP16 el único redondeo a 16 bits ocurre en
    // la salida final, igual que en la entrada.
    for (int i = 0; i < n; ++i) {
        y[i] = static_cast<T>(e[i] / s);
    }
}

// Referencia en double con std::exp
// FP64 tiene ~16 dígitos significativos frente a ~7 de FP32 y ~3 de FP16, así
// que su propio error es despreciable y sirve como "valor verdadero" para medir
// el error de las otras precisiones.
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
static int run(const char *prec, int n, int R) {
    // Datos: siempre el mismo vector en double
    // Semilla fija: CPU, GPU y FPGA generan exactamente el mismo vector y los
    // resultados de las tres plataformas son comparables.
    std::mt19937 gen(22);
    std::uniform_real_distribution<double> dist(-4.0, 4.0);
    std::vector<double> xd(n);
    for (int i = 0; i < n; ++i) xd[i] = dist(gen);

    // Referencia con el vector original en double (antes de convertir)
    // Así el error medido incluye también el redondeo de la entrada a T, que es
    // parte del costo real de usar FP16.
    std::vector<double> ref(n);
    softmax_ref(xd, ref);

    // Conversion a T
    std::vector<T> x(n), y(n);
    std::vector<float> e(n);
    for (int i = 0; i < n; ++i) x[i] = static_cast<T>(xd[i]);

    // ---------------- Verificacion ----------------
    // Una sola llamada fuera de la medición para comprobar que el resultado es
    // correcto antes de cronometrar.
    softmax(x.data(), y.data(), e.data(), n);

    // Se acumula en double para que la verificación no agregue error propio.
    // vmax/rmax inician en -1 porque toda salida de softmax es >= 0.
    double suma = 0.0, err_abs = 0.0, err_rel = 0.0;
    bool en_rango = true;
    int am = 0, am_ref = 0;
    double vmax = -1.0, rmax = -1.0;
    for (int i = 0; i < n; ++i) {
        // half_t -> float -> double: __fp16 se promueve primero a float.
        double v = static_cast<double>(static_cast<float>(y[i]));
        suma += v;
        if (v < 0.0 || v > 1.0) en_rango = false;
        if (v > vmax) { vmax = v; am = i; }
        if (ref[i] > rmax) { rmax = ref[i]; am_ref = i; }
        // El error relativo no tiene división entre cero: ref[i] > 0 siempre.
        double ea = std::fabs(v - ref[i]);
        double er = ea / ref[i];
        if (ea > err_abs) err_abs = ea;
        if (er > err_rel) err_rel = er;
    }

    std::printf("formato=%s n=%d suma=%.8f en_rango=%s argmax=%d (ref=%d, %s) "
                "err_abs=%.3e err_rel=%.3e\n",
                prec, n, suma, en_rango ? "si" : "no", am, am_ref,
                am == am_ref ? "coincide" : "NO coincide", err_abs, err_rel);

    // ---------------- Medicion ----------------
    // Protocolo: un calentamiento descartado y luego 5 lotes de R llamadas.
    // Una sola llamada dura del orden de microsegundos, cerca de la resolución
    // y del costo del propio reloj, por eso se cronometra el lote entero y se
    // divide entre R.
    // Escribir en una variable volatile obliga al compilador a ejecutar cada
    // llamada: sin esto, con -O3 podría detectar que el resultado no se usa y
    // eliminar o fusionar las llamadas del bucle.
    volatile float sink = 0.0f;  // usa el resultado de cada llamada

    // Calentamiento (descartado)
    // Carga datos e instrucciones en caché, resuelve fallos de página y da
    // tiempo a que el gobernador de frecuencia de la Jetson suba el reloj.
    for (int r = 0; r < R; ++r) {
        softmax(x.data(), y.data(), e.data(), n);
        sink = sink + static_cast<float>(y[r % n]);
    }

    // 5 lotes independientes para ver la variabilidad entre mediciones.
    const int M = 5;
    std::vector<double> t(M);
    for (int k = 0; k < M; ++k) {
        // steady_clock es monótono: no salta si se ajusta la hora del sistema.
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < R; ++r) {
            softmax(x.data(), y.data(), e.data(), n);
            sink = sink + static_cast<float>(y[r % n]);
        }
        auto t1 = std::chrono::steady_clock::now();
        t[k] = std::chrono::duration<double, std::micro>(t1 - t0).count() / R;
        std::printf("medicion %d: %.4f us/llamada\n", k + 1, t[k]);
    }

    // Se ordena una copia para no alterar el orden de t que va al CSV.
    // La mediana es robusta ante un lote afectado por una interrupción del SO.
    std::vector<double> ts = t;
    std::sort(ts.begin(), ts.end());
    std::printf("min=%.4f us  mediana=%.4f us  max=%.4f us  (R=%d)\n",
                ts[0], ts[M / 2], ts[M - 1], R);

    // ---------------- CSV ----------------
    // Se comprueba si el archivo ya existe para escribir el encabezado una sola
    // vez; luego se abre en modo append para acumular varias corridas.
    FILE *chk = std::fopen("cpu_results.csv", "r");
    bool existe = (chk != nullptr);
    if (chk) std::fclose(chk);

    FILE *f = std::fopen("cpu_results.csv", "a");
    if (!f) {
        std::fprintf(stderr, "No se pudo abrir cpu_results.csv\n");
        return 1;
    }
    if (!existe) std::fprintf(f, "precision,n,reps,us_por_llamada,err_abs,err_rel\n");
    for (int k = 0; k < M; ++k) {
        std::fprintf(f, "%s,%d,%d,%.6f,%.6e,%.6e\n", prec, n, R, t[k], err_abs, err_rel);
    }
    std::fclose(f);

    return 0;
}

int main(int argc, char **argv) {
    // Valores por defecto si no se pasan banderas.
    std::string prec = "fp32";
    int n = 100, R = 10000;
    // Las banderas vienen en pares (--bandera valor); una desconocida invalida
    // la precisión para que se imprima el uso.
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "--precision") prec = argv[i + 1];
        else if (a == "--n") n = std::atoi(argv[i + 1]);
        else if (a == "--reps") R = std::atoi(argv[i + 1]);
        else { prec = "error"; break; }
    }

    // n <= 1024 por especificación del laboratorio (mismo límite que la GPU).
    if ((prec != "fp32" && prec != "fp16") || n < 1 || n > 1024 || R < 1) {
        std::fprintf(stderr, "Uso: %s --precision fp32|fp16 --n N(<=1024) --reps R\n", argv[0]);
        return 1;
    }

    // La precisión se elige en tiempo de ejecución, pero cada rama instancia la
    // plantilla con un tipo distinto en tiempo de compilación.
    if (prec == "fp16") return run<half_t>("fp16", n, R);
    return run<float>("fp32", n, R);
}