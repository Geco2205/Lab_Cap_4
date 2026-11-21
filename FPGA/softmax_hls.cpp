#include "hls_math.h"

#ifdef SOFTMAX_FP16
#include "hls_half.h"
typedef half data_t;
#else
typedef float data_t;
#endif

// Softmax estable: x/y en data_t; operaciones internas en float.
// Version secuencial, sin PIPELINE, ARRAY_PARTITION ni DATAFLOW.
void softmax_hls(const data_t x[1000], data_t y[1000], int n) {
#pragma HLS INTERFACE mode=m_axi port=x offset=slave bundle=gmem0 depth=1000
#pragma HLS INTERFACE mode=m_axi port=y offset=slave bundle=gmem1 depth=1000
#pragma HLS INTERFACE mode=s_axilite port=x bundle=control
#pragma HLS INTERFACE mode=s_axilite port=y bundle=control
#pragma HLS INTERFACE mode=s_axilite port=n bundle=control
#pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    if (n < 1 || n > 1000) return;

    float e[1000];

    // Paso 1: reduccion del maximo.
    float m = static_cast<float>(x[0]);
MAX_LOOP:
    for (int i = 1; i < n; ++i) {
        float v = static_cast<float>(x[i]);
        if (v > m) m = v;
    }

    // Paso 2: exponenciales y reduccion de la suma.
    float s = 0.0f;
EXP_SUM_LOOP:
    for (int i = 0; i < n; ++i) {
        float v = hls::exp(static_cast<float>(x[i]) - m);
        e[i] = v;
        s += v;
    }

    // Paso 3: normalizacion.
NORMALIZE_LOOP:
    for (int i = 0; i < n; ++i) {
        y[i] = static_cast<data_t>(e[i] / s);
    }
}
