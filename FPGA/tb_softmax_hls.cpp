#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#ifdef SOFTMAX_FP16
#include "hls_half.h"
typedef half data_t;
#else
typedef float data_t;
#endif

void softmax_hls(const data_t x[1000], data_t y[1000], int n);

// Referencia en double, igual que en las versiones CPU/CUDA.
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

int main() {
    const int sizes[] = {10, 100, 500, 1000};
#ifdef SOFTMAX_FP16
    const char *precision = "fp16_mixto";
    const double tol_abs = 1.0e-3;
    const double tol_sum = 1.0e-2;
#else
    const char *precision = "fp32";
    const double tol_abs = 2.0e-5;
    const double tol_sum = 1.0e-4;
#endif

    static data_t x[1000], y[1000];
    int failures = 0;

    for (int test = 0; test < 4; ++test) {
        const int n = sizes[test];
        std::mt19937 gen(22);
        std::uniform_real_distribution<double> dist(-4.0, 4.0);

        std::vector<double> xd(n), ref(n);
        for (int i = 0; i < n; ++i) {
            xd[i] = dist(gen);
            x[i] = static_cast<data_t>(xd[i]);
            y[i] = static_cast<data_t>(0.0f);
        }
        softmax_ref(xd, ref);
        softmax_hls(x, y, n);

        double suma = 0.0, err_abs = 0.0, err_rel = 0.0;
        double max_y = -1.0, max_ref = -1.0;
        int am = 0, am_ref = 0;
        bool en_rango = true;

        for (int i = 0; i < n; ++i) {
            const double v = static_cast<double>(static_cast<float>(y[i]));
            suma += v;
            if (!std::isfinite(v) || v < 0.0 || v > 1.0) en_rango = false;
            if (v > max_y) { max_y = v; am = i; }
            if (ref[i] > max_ref) { max_ref = ref[i]; am_ref = i; }
            const double ea = std::fabs(v - ref[i]);
            const double er = ea / ref[i];
            err_abs = std::max(err_abs, ea);
            err_rel = std::max(err_rel, er);
        }

        const bool ok = en_rango && am == am_ref &&
                        std::fabs(suma - 1.0) <= tol_sum &&
                        err_abs <= tol_abs;
        if (!ok) ++failures;

        std::printf("[%s] N=%4d suma=%.8f rango=%s argmax=%s "
                    "err_abs=%.3e err_rel=%.3e %s\n",
                    precision, n, suma, en_rango ? "OK" : "FAIL",
                    am == am_ref ? "OK" : "FAIL",
                    err_abs, err_rel, ok ? "PASS" : "FAIL");
    }
    return failures == 0 ? 0 : 1;
}
