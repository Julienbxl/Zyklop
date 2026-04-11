// AUTO-GENERATED — do not edit
// test_msm_g2.cu — unit tests for msm_g2.cuh
// Compile: nvcc -O3 -std=c++17 -arch=sm_120 -I../include test_msm_g2.cu -o test_msm_g2
// Run:     ./test_msm_g2

#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <cuda_runtime.h>
#include "fp_bn254.cuh"
#include "fp2_bn254.cuh"
#include "msm_g2.cuh"

static bool eq8(const uint64_t a[8],const uint64_t b[8]){
    for(int i=0;i<8;i++) if(a[i]!=b[i]) return false; return true;}
static void print8(const char* t,const uint64_t x[8]){
    printf("%s [%016" PRIx64 " %016" PRIx64 " %016" PRIx64 " %016" PRIx64
                 " | %016" PRIx64 " %016" PRIx64 " %016" PRIx64 " %016" PRIx64 "]\n",
           t,x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]);
}

int main() {
    printf("[msm_g2] Running 12 hardcoded test vectors\n");
    fflush(stdout);
    int pass=0, fail=0;

    { // n1_rand_0 N=1
        const int N=1;
        uint64_t scalars[1*4] = {0x1e702bbb421c8496ULL,0x535107a7f4e697e3ULL,0x2cad940af510832cULL,0x081fed9bbd6e773dULL};
        uint64_t Px[1*8] = {0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL};
        uint64_t Py[1*8] = {0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL};
        bool exp_inf = false;
        uint64_t exp_x[8] = {0x73cdb82203be1a04ULL,0x4b9ee55a03e2b9d4ULL,0x95a613885ff7c51cULL,0x189816d1ddb1db4aULL,0x489d3c3364f98ba1ULL,0xcb6a22d0ac4fa368ULL,0xe50156cbd17957e5ULL,0x03f4299ca3d61cb4ULL};
        uint64_t exp_y[8] = {0x7d73047e2f6d4e63ULL,0x16b6f2a30cef25feULL,0x7b8467e4d1f2e433ULL,0x22cbc9e7f15efd07ULL,0xf22364e33aaa3a32ULL,0x5a1778b8c8179257ULL,0xe2b86852e6289b33ULL,0x1b3598aa8a566cdcULL};
        Fp2PointAff result; memset(&result,0,sizeof(result));
        msmG2(scalars,Px,Py,N,result);
        bool ok = exp_inf ? result.infinity : (!result.infinity && eq8(result.X,exp_x) && eq8(result.Y,exp_y));
        if(ok){ printf("  [PASS] n1_rand_0\n"); pass++; }
        else{
            printf("  [FAIL] n1_rand_0\n"); fail++;
            if(exp_inf) printf("    expected infinity, got inf=%d\n",(int)result.infinity);
            else{ print8("  got X",result.X); print8("  exp X",exp_x);
                   print8("  got Y",result.Y); print8("  exp Y",exp_y); }
        }
    }

    { // n1_rand_1 N=1
        const int N=1;
        uint64_t scalars[1*4] = {0xab2c79b3aeeae2b2ULL,0x6cc818c18fded2abULL,0xee3de8fc8b7cc4e2ULL,0x26b5938694f24730ULL};
        uint64_t Px[1*8] = {0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL};
        uint64_t Py[1*8] = {0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL};
        bool exp_inf = false;
        uint64_t exp_x[8] = {0x7c96ec9c5357798eULL,0xe7e007b0d7067d1bULL,0x26c8a14eee86bceeULL,0x19d1d866b7f6e80aULL,0x246b7c75104ef9c7ULL,0x0ca96e36ee50335eULL,0x4a4977d257b49213ULL,0x121fc2ee3453a886ULL};
        uint64_t exp_y[8] = {0x94a393e4d212abaeULL,0x5bcb165e8e018cdcULL,0x7279d136ce5ad1a8ULL,0x2ea9a9e687a213c4ULL,0x5d34f932d81baa70ULL,0x77669c216ecce7ccULL,0x5140f5e54334808bULL,0x1028a1517e08b2faULL};
        Fp2PointAff result; memset(&result,0,sizeof(result));
        msmG2(scalars,Px,Py,N,result);
        bool ok = exp_inf ? result.infinity : (!result.infinity && eq8(result.X,exp_x) && eq8(result.Y,exp_y));
        if(ok){ printf("  [PASS] n1_rand_1\n"); pass++; }
        else{
            printf("  [FAIL] n1_rand_1\n"); fail++;
            if(exp_inf) printf("    expected infinity, got inf=%d\n",(int)result.infinity);
            else{ print8("  got X",result.X); print8("  exp X",exp_x);
                   print8("  got Y",result.Y); print8("  exp Y",exp_y); }
        }
    }

    { // n1_rand_2 N=1
        const int N=1;
        uint64_t scalars[1*4] = {0x57b09ef0ea01f4b0ULL,0x33ba9763a32387fbULL,0x45ed91a70c638dfbULL,0x05fd24bdd56843cdULL};
        uint64_t Px[1*8] = {0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL};
        uint64_t Py[1*8] = {0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL};
        bool exp_inf = false;
        uint64_t exp_x[8] = {0x742ce107df79c798ULL,0x233a8e12af880f55ULL,0x128b8fd817d0da82ULL,0x0fad760e8d2c0b19ULL,0xb201ca1f802f30d9ULL,0xc735ec30177eac55ULL,0xd4bb4b08dd1c1bf6ULL,0x2c5241d5cc850ae5ULL};
        uint64_t exp_y[8] = {0x386404b2b63b7d8aULL,0xe2adc99d6a7cd9a0ULL,0xe9f412f4e67ab350ULL,0x1b249774f9fade02ULL,0xe178d498f4666646ULL,0xf7751b0b8de0b71dULL,0x9978210206e5038fULL,0x223c6e5b58683da6ULL};
        Fp2PointAff result; memset(&result,0,sizeof(result));
        msmG2(scalars,Px,Py,N,result);
        bool ok = exp_inf ? result.infinity : (!result.infinity && eq8(result.X,exp_x) && eq8(result.Y,exp_y));
        if(ok){ printf("  [PASS] n1_rand_2\n"); pass++; }
        else{
            printf("  [FAIL] n1_rand_2\n"); fail++;
            if(exp_inf) printf("    expected infinity, got inf=%d\n",(int)result.infinity);
            else{ print8("  got X",result.X); print8("  exp X",exp_x);
                   print8("  got Y",result.Y); print8("  exp Y",exp_y); }
        }
    }

    { // n1_scalar1 N=1
        const int N=1;
        uint64_t scalars[1*4] = {0x0000000000000001ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL};
        uint64_t Px[1*8] = {0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL};
        uint64_t Py[1*8] = {0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL};
        bool exp_inf = false;
        uint64_t exp_x[8] = {0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL};
        uint64_t exp_y[8] = {0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL};
        Fp2PointAff result; memset(&result,0,sizeof(result));
        msmG2(scalars,Px,Py,N,result);
        bool ok = exp_inf ? result.infinity : (!result.infinity && eq8(result.X,exp_x) && eq8(result.Y,exp_y));
        if(ok){ printf("  [PASS] n1_scalar1\n"); pass++; }
        else{
            printf("  [FAIL] n1_scalar1\n"); fail++;
            if(exp_inf) printf("    expected infinity, got inf=%d\n",(int)result.infinity);
            else{ print8("  got X",result.X); print8("  exp X",exp_x);
                   print8("  got Y",result.Y); print8("  exp Y",exp_y); }
        }
    }

    { // n1_scalar0 N=1
        const int N=1;
        uint64_t scalars[1*4] = {0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL};
        uint64_t Px[1*8] = {0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL};
        uint64_t Py[1*8] = {0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL};
        bool exp_inf = true;
        uint64_t exp_x[8] = {0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL};
        uint64_t exp_y[8] = {0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL};
        Fp2PointAff result; memset(&result,0,sizeof(result));
        msmG2(scalars,Px,Py,N,result);
        bool ok = exp_inf ? result.infinity : (!result.infinity && eq8(result.X,exp_x) && eq8(result.Y,exp_y));
        if(ok){ printf("  [PASS] n1_scalar0\n"); pass++; }
        else{
            printf("  [FAIL] n1_scalar0\n"); fail++;
            if(exp_inf) printf("    expected infinity, got inf=%d\n",(int)result.infinity);
            else{ print8("  got X",result.X); print8("  exp X",exp_x);
                   print8("  got Y",result.Y); print8("  exp Y",exp_y); }
        }
    }

    { // n2_ones N=2
        const int N=2;
        uint64_t scalars[2*4] = {0x0000000000000001ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000001ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL};
        uint64_t Px[2*8] = {0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL,0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL};
        uint64_t Py[2*8] = {0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL,0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL};
        bool exp_inf = false;
        uint64_t exp_x[8] = {0x42d3ae372af7a579ULL,0xd2f609cbc0a293b5ULL,0x57c1c76166f89cedULL,0x2ee858391ef2cc42ULL,0xa856e093920a72b8ULL,0x63806a546d033f3eULL,0x1e7eeefa55543decULL,0x056ca0f5743c0a90ULL};
        uint64_t exp_y[8] = {0xbcd0a4b05c092587ULL,0x5977f684ad4ff6baULL,0x4fde1ced78e96a59ULL,0x02750f99cf18d82eULL,0x332573d63fab5b02ULL,0xa83429e5be54f062ULL,0xaee376c3d20111f4ULL,0x0e103b8b9272ef7dULL};
        Fp2PointAff result; memset(&result,0,sizeof(result));
        msmG2(scalars,Px,Py,N,result);
        bool ok = exp_inf ? result.infinity : (!result.infinity && eq8(result.X,exp_x) && eq8(result.Y,exp_y));
        if(ok){ printf("  [PASS] n2_ones\n"); pass++; }
        else{
            printf("  [FAIL] n2_ones\n"); fail++;
            if(exp_inf) printf("    expected infinity, got inf=%d\n",(int)result.infinity);
            else{ print8("  got X",result.X); print8("  exp X",exp_x);
                   print8("  got Y",result.Y); print8("  exp Y",exp_y); }
        }
    }

    { // n2_rand_0 N=2
        const int N=2;
        uint64_t scalars[2*4] = {0x31a27fb6ebfcf519ULL,0xb22f21c34c0f3981ULL,0xb81885e0eccd7e35ULL,0x114e6cf24ead1710ULL,0x3e005cbda62fc9bbULL,0x1645241f9e863d98ULL,0xd10ca4eea60b4791ULL,0x27f730b197253c9cULL};
        uint64_t Px[2*8] = {0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL,0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL};
        uint64_t Py[2*8] = {0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL,0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL};
        bool exp_inf = false;
        uint64_t exp_x[8] = {0x70c3a2eb55cbcdbbULL,0x74b06399f2421122ULL,0xb51d27cb1233700eULL,0x14e2b3292cc1eb37ULL,0x213c2f8ef1c8a03eULL,0x9b293aca2602472bULL,0x4c6fa121c984ebacULL,0x212d0d3ebe269091ULL};
        uint64_t exp_y[8] = {0x67a248aaa3c9464eULL,0x98fdb04b1f64951fULL,0x1d56180ac7c1e375ULL,0x2f04ce3348e8f48aULL,0x36ee254f5eef5f20ULL,0x2b69d1103e25d3eeULL,0x3c9a1b1455a490f9ULL,0x062aea8fa0e9404eULL};
        Fp2PointAff result; memset(&result,0,sizeof(result));
        msmG2(scalars,Px,Py,N,result);
        bool ok = exp_inf ? result.infinity : (!result.infinity && eq8(result.X,exp_x) && eq8(result.Y,exp_y));
        if(ok){ printf("  [PASS] n2_rand_0\n"); pass++; }
        else{
            printf("  [FAIL] n2_rand_0\n"); fail++;
            if(exp_inf) printf("    expected infinity, got inf=%d\n",(int)result.infinity);
            else{ print8("  got X",result.X); print8("  exp X",exp_x);
                   print8("  got Y",result.Y); print8("  exp Y",exp_y); }
        }
    }

    { // n2_rand_1 N=2
        const int N=2;
        uint64_t scalars[2*4] = {0x92b86c4425602672ULL,0x231ddcfd59ba628aULL,0xbbd7021e1d9f39efULL,0x042dcf2d028a9eddULL,0x471555468fadb586ULL,0x68a2e19f3074c0a5ULL,0x1c882186c75f6331ULL,0x26b2572052b51d57ULL};
        uint64_t Px[2*8] = {0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL,0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL};
        uint64_t Py[2*8] = {0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL,0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL};
        bool exp_inf = false;
        uint64_t exp_x[8] = {0xf0e80c32e935a9fdULL,0x2d2e7e39b35e73e6ULL,0xaf4a9d61d2c71462ULL,0x0f0f3337e43c94a1ULL,0x1bb0dd49b566dfc4ULL,0x1a1f63a313e40542ULL,0x15d63f6051bc9e58ULL,0x17fb8f4a673de012ULL};
        uint64_t exp_y[8] = {0x1bf5373b1ff31296ULL,0x9203db7aa81c32e5ULL,0x29a0b084a7ef7209ULL,0x033a1fc26eaab6e4ULL,0x31c16a59aea6d748ULL,0x53923af4fa228eceULL,0x4c4b38ad70b48b1bULL,0x1af41628f1be52cdULL};
        Fp2PointAff result; memset(&result,0,sizeof(result));
        msmG2(scalars,Px,Py,N,result);
        bool ok = exp_inf ? result.infinity : (!result.infinity && eq8(result.X,exp_x) && eq8(result.Y,exp_y));
        if(ok){ printf("  [PASS] n2_rand_1\n"); pass++; }
        else{
            printf("  [FAIL] n2_rand_1\n"); fail++;
            if(exp_inf) printf("    expected infinity, got inf=%d\n",(int)result.infinity);
            else{ print8("  got X",result.X); print8("  exp X",exp_x);
                   print8("  got Y",result.Y); print8("  exp Y",exp_y); }
        }
    }

    { // n4_rand_0 N=4
        const int N=4;
        uint64_t scalars[4*4] = {0xf81449e7abff681cULL,0x98dc4a81f4cd11e1ULL,0xb9f1efc5fb9f9038ULL,0x2dc97b6b36399fb4ULL,0x5e7c9cbd24029479ULL,0xa45a0fe9320ad24bULL,0x2234dee8c3801f10ULL,0x16e72a43cc3713f5ULL,0xf7a34d683b95d740ULL,0xc55c694b38fa3df3ULL,0x7f592887bec59659ULL,0x0b15d132c36d5b3bULL,0x25222b2a6df1226aULL,0x0c61c38e76c41691ULL,0xd8205d19f8245f15ULL,0x1db7e9012bf47ddaULL};
        uint64_t Px[4*8] = {0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL,0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL,0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL,0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL};
        uint64_t Py[4*8] = {0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL,0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL,0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL,0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL};
        bool exp_inf = false;
        uint64_t exp_x[8] = {0x0d0622f22cb4dbafULL,0x5bd62b781d0c4b30ULL,0xbf894b007d56b746ULL,0x299078a90f649cb6ULL,0x43150d16bdaef4a4ULL,0x44738dbe6e566dc5ULL,0x951ee6b859484d5aULL,0x08789250386264f3ULL};
        uint64_t exp_y[8] = {0x881f2b388128d415ULL,0x5b725e4ccff19226ULL,0x787f4ee5daf2426bULL,0x24536a786f1ef673ULL,0xe74557c287f2fe68ULL,0x59467374d4ab52adULL,0xf18ee01dcd914700ULL,0x1432c3db2f447a23ULL};
        Fp2PointAff result; memset(&result,0,sizeof(result));
        msmG2(scalars,Px,Py,N,result);
        bool ok = exp_inf ? result.infinity : (!result.infinity && eq8(result.X,exp_x) && eq8(result.Y,exp_y));
        if(ok){ printf("  [PASS] n4_rand_0\n"); pass++; }
        else{
            printf("  [FAIL] n4_rand_0\n"); fail++;
            if(exp_inf) printf("    expected infinity, got inf=%d\n",(int)result.infinity);
            else{ print8("  got X",result.X); print8("  exp X",exp_x);
                   print8("  got Y",result.Y); print8("  exp Y",exp_y); }
        }
    }

    { // n4_rand_1 N=4
        const int N=4;
        uint64_t scalars[4*4] = {0xe77576d18542374bULL,0x0ced9dd99354702cULL,0x0c7e01d11844a56aULL,0x28a7430cb081371eULL,0xef834ea4986a78e9ULL,0xe8e909baaa5c5b8aULL,0x5195591edf33ff8eULL,0x13548b10992d8924ULL,0xe76d68c998c126a3ULL,0xcc7c80ab745c78faULL,0x2bed6787d2f45682ULL,0x1231d7138e67b540ULL,0x9a71d4e2cc9039a5ULL,0x53d99fac7313d7f7ULL,0x0b71f12f737a721eULL,0x1ce3a53c0aebbb52ULL};
        uint64_t Px[4*8] = {0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL,0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL,0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL,0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL};
        uint64_t Py[4*8] = {0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL,0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL,0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL,0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL};
        bool exp_inf = false;
        uint64_t exp_x[8] = {0x8596dff84248d440ULL,0x3f2f0f06decb1e54ULL,0xcc36de710c4fc580ULL,0x00c6a714751abfd3ULL,0x8841a57ddbbbe850ULL,0x9ee89e44fb96221bULL,0x5fd7c4f12c4b0bc1ULL,0x0dcd585fd0b8f26bULL};
        uint64_t exp_y[8] = {0xc2371316760ede4aULL,0x56a42ddd90f33be3ULL,0x21d1a6a81b446888ULL,0x1cc7190f15242e42ULL,0x90d615512927fcbeULL,0x68ab74924c5ce34eULL,0x7bf14de7446cf0aaULL,0x04949e5d07d35533ULL};
        Fp2PointAff result; memset(&result,0,sizeof(result));
        msmG2(scalars,Px,Py,N,result);
        bool ok = exp_inf ? result.infinity : (!result.infinity && eq8(result.X,exp_x) && eq8(result.Y,exp_y));
        if(ok){ printf("  [PASS] n4_rand_1\n"); pass++; }
        else{
            printf("  [FAIL] n4_rand_1\n"); fail++;
            if(exp_inf) printf("    expected infinity, got inf=%d\n",(int)result.infinity);
            else{ print8("  got X",result.X); print8("  exp X",exp_x);
                   print8("  got Y",result.Y); print8("  exp Y",exp_y); }
        }
    }

    { // n8_rand_0 N=8
        const int N=8;
        uint64_t scalars[8*4] = {0x5b7db528a76df41dULL,0x8e4d589ff4f05857ULL,0xbd0b7aee9c152f78ULL,0x19d9fb83b796a5deULL,0x3b67bd0b40e6a77cULL,0xe39546f7f1567445ULL,0xff0f7770a3c4a8eaULL,0x2a3b04e10de7d796ULL,0xcf17a176bf31ab1cULL,0x84d348bcd71c489dULL,0x89da01009e8ea171ULL,0x1ec60486098b87b0ULL,0x1b97408fc36b0dadULL,0x8bdb79c19e2cd738ULL,0x6e8155e57e34c04bULL,0x2b701217bee65115ULL,0xa1889d7bf9d7687dULL,0xd5ef93a819d66345ULL,0xb2b6c1ed9994ddddULL,0x1ff7fbcc1dda9e10ULL,0xd3f23c7a44f0c614ULL,0xf5774d04794a1834ULL,0x365ee743da25777cULL,0x0b968888528b6af9ULL,0x16263890a1d46900ULL,0x2276b67ef3cc02f8ULL,0xe5cd1462c3345bdcULL,0x04c3c66b3ec9c1a0ULL,0x034e9e25cc99f46aULL,0x1bd7080ad6512723ULL,0x6b2f49430ad52e82ULL,0x12ceb254d6691232ULL};
        uint64_t Px[8*8] = {0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL,0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL,0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL,0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL,0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL,0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL,0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL,0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL};
        uint64_t Py[8*8] = {0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL,0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL,0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL,0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL,0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL,0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL,0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL,0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL};
        bool exp_inf = false;
        uint64_t exp_x[8] = {0xf70b365fcd2132e1ULL,0xd591a05b37e94f42ULL,0xcf3a1ec79f5bf081ULL,0x261890ee11af71abULL,0x3b9f5b764e2f741aULL,0xb7447de6c7cf7cabULL,0x5227509a29f6505eULL,0x00b1fe6771cebb6aULL};
        uint64_t exp_y[8] = {0xeacf0e38362f3898ULL,0x432ba6a861946adcULL,0x6d589901c19a0575ULL,0x18f3b2d46cabae76ULL,0x015012e4abdbd8f5ULL,0xbb2f934ff06c4ea2ULL,0x51fea751097f692dULL,0x2393a98a38ace35dULL};
        Fp2PointAff result; memset(&result,0,sizeof(result));
        msmG2(scalars,Px,Py,N,result);
        bool ok = exp_inf ? result.infinity : (!result.infinity && eq8(result.X,exp_x) && eq8(result.Y,exp_y));
        if(ok){ printf("  [PASS] n8_rand_0\n"); pass++; }
        else{
            printf("  [FAIL] n8_rand_0\n"); fail++;
            if(exp_inf) printf("    expected infinity, got inf=%d\n",(int)result.infinity);
            else{ print8("  got X",result.X); print8("  exp X",exp_x);
                   print8("  got Y",result.Y); print8("  exp Y",exp_y); }
        }
    }

    { // n8_rand_1 N=8
        const int N=8;
        uint64_t scalars[8*4] = {0x8f3aa0ca20571f84ULL,0xded49449406f8824ULL,0x65623c7fd0de7eedULL,0x098d65060fc9c03bULL,0x6f7092a1e1d88fa5ULL,0x172b29faf11a9ca3ULL,0xbb06a7b51624d279ULL,0x1cf0a0167089b646ULL,0x4237ca6b5890256cULL,0x23ffb80c45700388ULL,0xcc72b24235b1a9f9ULL,0x27ad6a2970bc0e5fULL,0xfde590b8faa50906ULL,0x859131396f6594cfULL,0xe1c90be6e3a9ebe5ULL,0x0502259244d386daULL,0x999f4c727d2181b5ULL,0xc1744107318735afULL,0x372f15717b52e80eULL,0x11b80cef2b1d4a06ULL,0xe6f61ae7385f5093ULL,0xe11b24c46050c7b3ULL,0x41c1c8a54891e0a3ULL,0x01c4b4b6bd108615ULL,0xbcfb9c1dc4db2d47ULL,0xefba03f470a3d0ccULL,0x9ce5ae2badc85a88ULL,0x219a7ce717503016ULL,0xc1a8153171399a18ULL,0x300b515747b64c78ULL,0xcdcbce5ea585f5b6ULL,0x297fb629576daa6aULL};
        uint64_t Px[8*8] = {0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL,0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL,0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL,0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL,0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL,0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL,0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL,0x8e83b5d102bc2026ULL,0xdceb1935497b0172ULL,0xfbb8264797811adfULL,0x19573841af96503bULL,0xafb4737da84c6140ULL,0x6043dd5a5802d8c4ULL,0x09e950fc52a02f86ULL,0x14fef0833aea7b6bULL};
        uint64_t Py[8*8] = {0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL,0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL,0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL,0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL,0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL,0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL,0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL,0x619dfa9d886be9f6ULL,0xfe7fd297f59e9b78ULL,0xff9e1a62231b7dfeULL,0x28fd7eebae9e4206ULL,0x64095b56c71856eeULL,0xdc57f922327d3cbbULL,0x55f935be33351076ULL,0x0da4a0e693fd6482ULL};
        bool exp_inf = false;
        uint64_t exp_x[8] = {0xa27958375ee499c8ULL,0x80c0d8694ce3fba4ULL,0xe75bc3a1a963f6e9ULL,0x18019112e22a0522ULL,0xb5ded5e2e9ea3f3eULL,0xea09172eb90d7332ULL,0x7a8efa783c83a059ULL,0x04cb906db203336fULL};
        uint64_t exp_y[8] = {0x287aa845da3cb379ULL,0x459914bbe174a8ccULL,0x193df6a03fc17e9bULL,0x1ddad993d37afc46ULL,0x70749650ca10e2b1ULL,0x319021b9bc759df1ULL,0x4ab5c22d6bcdf033ULL,0x0dcd072e59cacd2eULL};
        Fp2PointAff result; memset(&result,0,sizeof(result));
        msmG2(scalars,Px,Py,N,result);
        bool ok = exp_inf ? result.infinity : (!result.infinity && eq8(result.X,exp_x) && eq8(result.Y,exp_y));
        if(ok){ printf("  [PASS] n8_rand_1\n"); pass++; }
        else{
            printf("  [FAIL] n8_rand_1\n"); fail++;
            if(exp_inf) printf("    expected infinity, got inf=%d\n",(int)result.infinity);
            else{ print8("  got X",result.X); print8("  exp X",exp_x);
                   print8("  got Y",result.Y); print8("  exp Y",exp_y); }
        }
    }

    printf("\n[msm_g2] %d/12 PASS\n",pass);
    return fail?1:0;
}