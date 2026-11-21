#include "hls_math.h"

#ifdef SOFTMAX_FP16
#include "hls_half.h"
typedef half data_t;
#else
typedef float data_t;
#endif

// ============================================================
// Etapa 1: lectura de entrada y busqueda del maximo.
// ============================================================

static void stage_max(
    const data_t x[1000],
    data_t x_cache[1000],
    int n,
    float &m
) {
#pragma HLS INLINE off

    data_t first = x[0];
    x_cache[0] = first;

    float max_value = static_cast<float>(first);

MAX_LOOP:
    for (int i = 1; i < n; ++i) {
        data_t xi = x[i];

        x_cache[i] = xi;

        float v = static_cast<float>(xi);

        if (v > max_value) {
            max_value = v;
        }
    }

    m = max_value;
}

// ============================================================
// Etapa 2: exponenciales y acumulacion.
// ============================================================

static void stage_exp_sum(
    const data_t x_cache[1000],
    float m,
    float e[1000],
    int n,
    float &s
) {
#pragma HLS INLINE off

    float sum = 0.0f;

EXP_SUM_LOOP:
    for (int i = 0; i < n; ++i) {

        float v = hls::exp(
            static_cast<float>(x_cache[i]) - m
        );

        e[i] = v;
        sum += v;
    }

    s = sum;
}

// ============================================================
// Etapa 3: normalizacion con PIPELINE.
// ============================================================

static void stage_normalize(
    const float e[1000],
    float s,
    data_t y[1000],
    int n
) {
#pragma HLS INLINE off

NORMALIZE_LOOP:
    for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1

        y[i] = static_cast<data_t>(e[i] / s);
    }
}

// ============================================================
// Region DATAFLOW: conexion entre tareas.
// ============================================================

static void softmax_dataflow_core(
    const data_t x[1000],
    data_t y[1000],
    int n
) {
#pragma HLS INLINE off

    data_t x_cache[1000];
    float e[1000];

    float m;
    float s;

#pragma HLS DATAFLOW

    stage_max(x, x_cache, n, m);

    stage_exp_sum(x_cache, m, e, n, s);

    stage_normalize(e, s, y, n);
}

// ============================================================
// Funcion principal: interfaces AXI.
// ============================================================

void softmax_hls(
    const data_t x[1000],
    data_t y[1000],
    int n
) {
#pragma HLS INTERFACE mode=m_axi port=x offset=slave bundle=gmem0 depth=1000
#pragma HLS INTERFACE mode=m_axi port=y offset=slave bundle=gmem1 depth=1000

#pragma HLS INTERFACE mode=s_axilite port=x bundle=control
#pragma HLS INTERFACE mode=s_axilite port=y bundle=control
#pragma HLS INTERFACE mode=s_axilite port=n bundle=control
#pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    if (n < 1 || n > 1000) {
        return;
    }

    softmax_dataflow_core(x, y, n);
}
