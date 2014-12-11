#include <string.h>
#include <math.h>
#include <omp.h>
#include "benchmark.h"
#include <stdio.h>
#include <emmintrin.h>

void eig(float *v, float *A, float *u, size_t n, unsigned iters) {
    /* TODO: write a faster version of eig */
    //float v2[n*n];
    

    for (size_t k = 0; k < iters; k += 1) {
        /* v_k = Au_{k-1} */
        memset(v, 0, n * n * sizeof(float));
	//memset(v2, 0, n * n * sizeof(float));
	/*
	  for (size_t l = 0; l < n; l++) {
	  for (size_t j = 0; j < n; j++) {
	  for (size_t i = 0; i < n; i++) {
	  v2[i + l*j] = u[j + l*j] * A[i + j*n];
	  }
	  }
	  }*/
        
	#pragma omp parallel for
	for (size_t l = 0; l < n; l++) {

	    
	    for (size_t j = 0; j < n/4*4; j+=4) {

	    	float* uoffset = u+j+l*n;
		
		__m128 uvect0  = _mm_load1_ps(uoffset + 0);
		__m128 uvect1  = _mm_load1_ps(uoffset + 1);
		__m128 uvect2  = _mm_load1_ps(uoffset + 2);
		__m128 uvect3  = _mm_load1_ps(uoffset + 3);

		for (size_t i = 0; i < n/16*16; i += 16) {
			float* voffset = v+i+l*n;

		    __m128 globvect0 = _mm_loadu_ps(voffset + 0);
		    __m128 globvect1 = _mm_loadu_ps(voffset + 4);
		    __m128 globvect2 = _mm_loadu_ps(voffset + 8);
		    __m128 globvect3 = _mm_loadu_ps(voffset + 12);

		    float* Ainit = A+i+n*j;
		    float* A0 = Ainit + n*0;
		    float* A1 = Ainit + n*1;
		    float* A2 = Ainit + n*2;
		    float* A3 = Ainit + n*3;
		    
		    globvect0 = _mm_add_ps(globvect0, _mm_mul_ps(_mm_loadu_ps(A0 + 0), uvect0));
		    globvect1 = _mm_add_ps(globvect1, _mm_mul_ps(_mm_loadu_ps(A0 + 4), uvect0));
		    globvect2 = _mm_add_ps(globvect2, _mm_mul_ps(_mm_loadu_ps(A0 + 8), uvect0));
		    globvect3 = _mm_add_ps(globvect3, _mm_mul_ps(_mm_loadu_ps(A0 + 12), uvect0));
		    
		    globvect0 = _mm_add_ps(globvect0, _mm_mul_ps(_mm_loadu_ps(A1 + 0), uvect1));
		    globvect1 = _mm_add_ps(globvect1, _mm_mul_ps(_mm_loadu_ps(A1 + 4), uvect1));
		    globvect2 = _mm_add_ps(globvect2, _mm_mul_ps(_mm_loadu_ps(A1 + 8), uvect1));
		    globvect3 = _mm_add_ps(globvect3, _mm_mul_ps(_mm_loadu_ps(A1 + 12), uvect1));
		    
		    globvect0 = _mm_add_ps(globvect0, _mm_mul_ps(_mm_loadu_ps(A2 + 0), uvect2));
		    globvect1 = _mm_add_ps(globvect1, _mm_mul_ps(_mm_loadu_ps(A2 + 4), uvect2));
		    globvect2 = _mm_add_ps(globvect2, _mm_mul_ps(_mm_loadu_ps(A2 + 8), uvect2));
		    globvect3 = _mm_add_ps(globvect3, _mm_mul_ps(_mm_loadu_ps(A2 + 12), uvect2));
		    
		    globvect0 = _mm_add_ps(globvect0, _mm_mul_ps(_mm_loadu_ps(A3 + 0), uvect3));
		    globvect1 = _mm_add_ps(globvect1, _mm_mul_ps(_mm_loadu_ps(A3 + 4), uvect3));
		    globvect2 = _mm_add_ps(globvect2, _mm_mul_ps(_mm_loadu_ps(A3 + 8), uvect3));
		    globvect3 = _mm_add_ps(globvect3, _mm_mul_ps(_mm_loadu_ps(A3 + 12), uvect3));
		    
		    _mm_storeu_ps(voffset + 0, globvect0);
		    _mm_storeu_ps(voffset + 4, globvect1);
		    _mm_storeu_ps(voffset + 8, globvect2);
		    _mm_storeu_ps(voffset + 12, globvect3);	
  
		}
		
		for (size_t i = n/16*16; i < n/4*4; i += 4) {
			float* voffset = v + i + l*n;

		    __m128 globvect = _mm_loadu_ps(voffset);

		    float* Ainit = A + i + n*j;

		    globvect = _mm_add_ps(globvect, _mm_mul_ps(_mm_loadu_ps(Ainit + n*0), uvect0));
		    globvect = _mm_add_ps(globvect, _mm_mul_ps(_mm_loadu_ps(Ainit + n*1), uvect1));
		    globvect = _mm_add_ps(globvect, _mm_mul_ps(_mm_loadu_ps(Ainit + n*2), uvect2));
		    globvect = _mm_add_ps(globvect, _mm_mul_ps(_mm_loadu_ps(Ainit + n*3), uvect3));

		    _mm_storeu_ps(voffset, globvect);
		    
		}

		
		
		float global0 = u[j + 0 + l*n];
		for (size_t i = n/4*4; i < n; i++) {
		    v[i + l*n] += global0 * A[i + n*(j + 0)];
		}

		float global1 = u[j + 1 + l*n];
		for (size_t i = n/4*4; i < n; i++) {
		    v[i + l*n] += global1 * A[i + n*(j + 1)];
		}

		float global2 = u[j + 2 + l*n];
		for (size_t i = n/4*4; i < n; i++) {
		    v[i + l*n] += global2 * A[i + n*(j + 2)];
		}

		float global3 = u[j + 3 + l*n];
		for (size_t i = n/4*4; i < n; i++) {
		    v[i + l*n] += global3 * A[i + n*(j + 3)];
		}
		
	
	    }

	   

	    for (size_t j = n/4*4; j < n; j++) {
		
		__m128 uvect  = _mm_load1_ps(u + j + l*n);
		for (size_t i = 0; i < n/16*16; i += 16) {
			float* voffset = v + i + l*n;

		    __m128 globvect0 = _mm_loadu_ps(voffset + 0);
		    __m128 globvect1 = _mm_loadu_ps(voffset + 4);
		    __m128 globvect2 = _mm_loadu_ps(voffset + 8);
		    __m128 globvect3 = _mm_loadu_ps(voffset + 12);

		    float* Ainit = A + i + n*j;

		    globvect0 = _mm_add_ps(globvect0, _mm_mul_ps(_mm_loadu_ps(Ainit + 0), uvect));
		    globvect1 = _mm_add_ps(globvect1, _mm_mul_ps(_mm_loadu_ps(Ainit + 4), uvect));
		    globvect2 = _mm_add_ps(globvect2, _mm_mul_ps(_mm_loadu_ps(Ainit + 8), uvect));
		    globvect3 = _mm_add_ps(globvect3, _mm_mul_ps(_mm_loadu_ps(Ainit + 12), uvect));

		    _mm_storeu_ps(voffset + 0, globvect0);
		    _mm_storeu_ps(voffset + 4, globvect1);
		    _mm_storeu_ps(voffset + 8, globvect2);
		    _mm_storeu_ps(voffset + 12, globvect3);
		    
		}
		
		for (size_t i = n/16*16; i < n/4*4; i += 4) {
		    __m128 globvect = _mm_loadu_ps(v + i + l*n);

		    __m128 avect = _mm_loadu_ps(A + i + n*j);
		    avect = _mm_mul_ps(avect, uvect);
		    globvect = _mm_add_ps(globvect, avect);

		    _mm_storeu_ps(v + i + l*n, globvect);
		    
		}

		
		
		float global = u[j + l*n];
		for (size_t i = n/4*4; i < n; i++) {
		    v[i + l*n] += global * A[i + n*j];
		}
	    }
	    //}
        

	    /* mu_k = ||v_k|| */
        
        
	    //#pragma omp parallel for 
	    //for (size_t l = 0; l < n; l += 1) {
            float muglob = 0;
            

            __m128 arr1, arr2, arr3, arr4, sum;
            sum = _mm_setzero_ps();
            size_t i;

            for (i = 0; i < n/16*16; i+=16) {
            	float* voffset = v + i + l*n;

		arr1 = _mm_loadu_ps( voffset + 0);
		arr2 = _mm_loadu_ps( voffset + 4);
		arr3 = _mm_loadu_ps( voffset + 8);
		arr4 = _mm_loadu_ps( voffset + 12);

		arr1 = _mm_mul_ps(arr1, arr1);
		arr2 = _mm_mul_ps(arr2, arr2);
		arr3 = _mm_mul_ps(arr3, arr3);
		arr4 = _mm_mul_ps(arr4, arr4);

		arr1 = _mm_add_ps(arr1, arr2);
		arr3 = _mm_add_ps(arr3, arr4);
		arr1 = _mm_add_ps(arr1, arr3);

		sum = _mm_add_ps(arr1, sum);
            
            }
            float temp[4];
            _mm_storeu_ps(temp, sum);

            muglob += temp[0] + temp[1] + temp[2] + temp[3];

            for (i = n/16*16; i < n/4*4; i+=4) {
                float v0 = v[i+0 + l*n];
                float v1 = v[i+1 + l*n];
                float v2 = v[i+2 + l*n];
                float v3 = v[i+3 + l*n];

                muglob += v0 * v0;
                muglob += v1 * v1;
                muglob += v2 * v2;
                muglob += v3 * v3;

            }
            for (i = n/4*4; i < n; i++) {
                muglob += v[i + l*n] * v[i + l*n];
            }
            
            muglob = sqrt(muglob);   
	    //}

	    /* u_k = v_k / mu_k */
	    //#pragma omp parallel for

	    //for (size_t l = 0; l < n; l+= 1) {
	    __m128 globvect = _mm_set1_ps(muglob);
	    for (size_t i = 0; i < n/4*4; i+=4) {
		
		__m128 vvect = _mm_loadu_ps(v + i + l*n);

		__m128 uvect = _mm_div_ps(vvect, globvect);
		_mm_storeu_ps(u + i + l*n, uvect);

	    }

	    for (size_t i = n/4*4; i < n; i++) {
		u[i + l*n] = v[i + l*n] / muglob;
	    }
		  
	}
    }
}
