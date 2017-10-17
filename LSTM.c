// ***************************************
// ************** LSTM *******************
// ***************************************
// f_t = sigmoid( w_fx * x_t + w_fh * h_tm1 + b_f)
// i_t = sigmoid( w_ix * x_t + w_ih * h_tm1 + b_i)
// c_wave_t = tanh( w_cx * x_t + w_ch * h_tm1 + b_c)
// o_t = sigmoid( w_ox * x_t + w_oh * h_tm1 + b_o)
// c_t = f_t * c_tm1 + i_t * c_wave_t
// h_t = o_t * tanh(c_t)
// References
// [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) (original 1997 paper)

#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include<sys/time.h>
#include <omp.h>
#include <stdbool.h>

//share global memory
float** A;
float** B;
float** C;
float* x_temp;
float* f_t;
float* i_t;
float* c_wave_t;
float* o_t;
float* c_t;

// This is a batch GEMM implementation of inference benchmark for Long short term memory unit.
// Method
// batch GEMM for w_x * x both inside every timestep and accross timesteps. batch GEMM for w_h * h inside every timestep.
// Arguments
// wx contains w_fx, w_ix, w_cx, w_ox, w_h contains w_fh, w_ih, w_ch, w_oh, b contains b_f, b_i, b_c, b_o, h_0 and c_0 are initial state of h and c respectively.
void  LSTM_batch_gemm(int batch_size, int time_step, int input_dim, int hid, float* w_x, float* w_h, float* b, float* x, float* h_0, float* c_0, /*out*/float* y, bool return_sequences){
    int i,j,p;
    // w_x * x
    MKL_INT m[1]; 
    MKL_INT n[1]; 
    MKL_INT k[1]; 
    
    MKL_INT lda[1]; 
    MKL_INT ldb[1]; 
    MKL_INT ldc[1]; 
    
    CBLAS_TRANSPOSE transA[1]; 
    CBLAS_TRANSPOSE transB[1]; 
    
    float alpha[1]; 
    float beta[1]; 
    MKL_INT size_per_grp[1]; 

    m[0] = hid;
    k[0] = input_dim;
    n[0] = batch_size;
    
    lda[0] = k[0]; 
    ldb[0] = n[0]; 
    ldc[0] = n[0]; 
    
    transB[0] = CblasNoTrans; 
    transA[0] = CblasNoTrans; 
    
    alpha[0] = 1.0; 
    if (b == NULL) {
        beta[0] = 0.0;
    }
    else {
        beta[0] = 1.0;
        #pragma omp parallel for 
        for (i = 0; i < time_step; i++) { 
            for (j = 0; j < batch_size; j++) { 
                for (p = 0; p < hid; p++) { 
                    size_t offset0 = i * batch_size * hid + j * hid + p; 
                    size_t offset1 = (i + time_step) * batch_size * hid + j * hid + p; 
                    size_t offset2 = (i + 2 * time_step) * batch_size * hid + j * hid + p; 
                    size_t offset3 = (i + 3 * time_step) * batch_size * hid + j * hid + p; 
        
                    x_temp[offset0] = b_ptr[p]; 
                    x_temp[offset1] = b_ptr[p + hid]; 
                    x_temp[offset2] = b_ptr[p + 2 * hid]; 
                    x_temp[offset2] = b_ptr[p + 3 * hid]; 
                } 
            } 
        } 

    }
    size_per_grp[0] = 4 * time_step;

    if (NULL == A || NULL == B || NULL == C || NULL == x_temp || NULL == f_t || NULL == i_t || NULL == c_wave_t || NULL == o_t || NULL == c_t) {
        printf( "\n ERROR: malloc global buffers failed \n\n");
        return;
    }
    #pragma omp parallel for 
    for (i = 0; i < time_step; i++) { 
        A[i] = w_x;                                       // w_fx
        A[i + time_step] = w_x + input_dim * hid;         // w_ix
        A[i + 2 * time_step] = w_x + 2 * input_dim * hid; // w_cx 
        A[i + 3 * time_step] = w_x + 3 * input_dim * hid; // w_ox 
    
        B[i] = x + i * k[0] * n[0]; 
        B[i + time_step] = B[i]; 
        B[i + 2 * time_step] = B[i]; 
        B[i + 3 * time_step] = B[i]; 
    
        C[i] = x_temp + i * m[0] * n[0]; 
        C[i + time_step] = x_temp + (i + time_step) * m[0] * n[0]; 
        C[i + 2 * time_step] = x_temp + (i + 2 * time_step) * m[0] * n[0]; 
        C[i + 3 * time_step] = x_temp + (i + 3 * time_step) * m[0] * n[0]; 
    } 
    cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp); 

    // loop on step
    m[0] = hid;
    k[0] = hid;
    n[0] = batch_size;
    
    beta[0] = 1.0;

    lda[0] = k[0]; 
    ldb[0] = n[0]; 
    ldc[0] = n[0]; 
    size_per_grp[0] = 4;
    
    A[0] = w_h;                //w_fh
    A[1] = w_h + hid * hid;    //w_ih
    A[2] = w_h + 2 * hid * hid;//w_ch
    A[3] = w_h + 3 * hid * hid;//w_oh
    
    B[0] = h_0;
    B[1] = h_0;
    B[2] = h_0;
    B[3] = h_0;

    size_t mn = m[0] * n[0];
    #pragma omp parallel for
    for (j = 0; j < mn; j++) {
        c_t[j] = c_0[j];
    }

    for (i = 0; i < time_step; i++) {
        // f,i,c_wave,o
        C[0] = x_temp + i * m[0] * n[0];
        C[1] = x_temp + (i + time_step) * m[0] * n[0];
        C[2] = x_temp + (i + 2 * time_step) * m[0] * n[0];
        C[3] = x_temp + (i + 3 * time_step) * m[0] * n[0];

        cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp);

        // sigmoid for f,i,o, tanh for c_wave
        #pragma omp parallel for
        for (j = 0; j < mn; j++) {
            float exp_f = exp((float)(C[0][j]));
            float exp_i = exp((float)(C[1][j]));
            c_wave_t[j] = tanh((float)(C[2][j]));
            float exp_o = exp((float)(C[3][j]));
            f_t[j] = exp_f / ((float)1.0 + exp_f);        
            i_t[j] = exp_i / ((float)1.0 + exp_i);
            o_t[j] = exp_o / ((float)1.0 + exp_o);
        }
        //c
        if (i > 0) {
            #pragma omp parallel for 
            for (j = 0; j < mn; j++) { 
                c_t[j] = (float)((float)(f_t[j]) * (float)(c_t[j]) + (float)(i_t[j]) * (float)(c_wave_t[j])); 
            }
        }
        //h
        float* y_ptr = NULL;
        if (return_sequences) {
            y_ptr = y + i * batch_size * hid;
        } else {
            y_ptr = y;
        }
        #pragma omp parallel for
        for (j = 0; j < mn; j++) {
            y_ptr[j] = (float)(o_t[j]) * tanh((float)(c_t[j]));
        }
        // update
        B[0] = y_ptr;
        B[1] = B[0];
        B[2] = B[0];
        B[3] = B[0];
        printf( "\n");
    }
}

void main() {
    srand(45678);
    int i,j;
    //reuse memory
    //assume timestep changes for diff input length
    int max_len = 128;//max timestep
    int batch_size = 2;
    int time_step = 3;
    int input_dim = 10;
    int hid = 10;

    A = (float**)mkl_malloc(4 * max_len * sizeof (float*), 64);
    B = (float**)mkl_malloc(4 * max_len * sizeof (float*), 64);
    C = (float**)mkl_malloc(4 * max_len * sizeof (float*), 64);
    x_temp = (float*)mkl_malloc(max_len * 4 * batch_size * hid * sizeof (float), 64);
    f_t = (float*)mkl_malloc(batch_size * hid * sizeof (float), 64);
    i_t = (float*)mkl_malloc(batch_size * hid * sizeof (float), 64);
    c_wave_t = (float*)mkl_malloc(batch_size * hid * sizeof (float), 64);
    o_t = (float*)mkl_malloc(batch_size * hid * sizeof (float), 64);
    c_t = (float*)mkl_malloc(batch_size * hid * sizeof (float), 64);

    bool return_sequences = false;
    float* w_x;
    float* w_h;
    float* b;
    float* x;
    float* h_0;
    float* c_0;
    float* y;
    w_x = (float*)mkl_malloc(4 * hid * input_dim * sizeof (float), 64);
    w_h = (float*)mkl_malloc(4 * hid * hid * sizeof (float), 64);
    b = NULL;
    x = (float*)mkl_malloc(time_step * input_dim * batch_size * sizeof (float), 64);
    h_0 = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64);
    c_0 = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64);
    y = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64);
    memset(y, 0, sizeof(float) * hid * batch_size);
    for (i = 0; i < 4 * hid * input_dim; i++) {
        w_x[i] = ((float)rand()/(float)RAND_MAX);
    }
    for (i = 0; i < 4 * hid * hid; i++) {
        w_h[i] = ((float)rand()/(float)RAND_MAX);
    }
    //for (i = 0; i < 4 * hid; i++) {
    //    b[i] = ((float)rand()/(float)RAND_MAX);
    //}
    for (i = 0; i < time_step * input_dim * batch_size; i++) {
        x[i] = ((float)rand()/(float)RAND_MAX * 2.0f - 1.0f);
    }
    for (i = 0; i < hid * batch_size; i++) {
        h_0[i] = ((float)rand()/(float)RAND_MAX);
    }
    for (i = 0; i < hid * batch_size; i++) {
        c_0[i] = ((float)rand()/(float)RAND_MAX);
    }

    LSTM_batch_gemm(batch_size, time_step, input_dim, hid, w_x, w_h, b, x, h_0, c_0, /*out*/y, return_sequences);       
    
    printf("output:\n");
    for (i = 0; i < hid * batch_size; i++) {
        printf( "%f ",y[i]);
    }
    printf( "\n");

    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
    mkl_free(x_temp);
    mkl_free(f_t);
    mkl_free(i_t);
    mkl_free(c_wave_t);
    mkl_free(o_t);
    mkl_free(c_t);
    mkl_free(w_x);
    mkl_free(w_h);
    mkl_free(x);
    mkl_free(h_0);
    mkl_free(c_0);
    mkl_free(y);
}
