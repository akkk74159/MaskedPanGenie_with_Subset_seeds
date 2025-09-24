#include "hmm_gpu.h"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <numeric>
#include <iostream>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t _e = (call);                                                 \
    if (_e != cudaSuccess)                                                   \
        throw std::runtime_error(cudaGetErrorString(_e));                    \
  } while (0)


// ----- DEBUG utilities -----
static std::vector<double> cpu_forward_reference(const std::vector<double>& prevF,
	const std::vector<double>& trans,
	const std::vector<double>& emis,
	int prevP, int P)
{
    int curSize = P*P;
    int prevSize = prevP*prevP;
    std::vector<double> cur(curSize, 0.0);

    if (prevSize == 0) {                 // 第一個 slice
        for (int i = 0; i < curSize; ++i) 
            cur[i] = emis[i];
        } 
    else {
        for (int i = 0; i < curSize; ++i) {
            double s = 0.0;
            for (int j = 0; j < prevSize; ++j)
                s += prevF[j] * trans[i * prevSize + j];
            cur[i] = s * emis[i];
        }
    }
    /* normalise (跟 GPU 做的一樣) */
    double tot = std::accumulate(cur.begin(), cur.end(), 0.0);
    if (tot == 0.0) 
        tot = 1.0 / curSize;
    for (double& v : cur) 
        v /= tot;
    return cur;
}

static void debug_diff(const std::vector<double>& cpu, const std::vector<double>& gpu, int P, int sliceId)
{
    const double eps = 1e-9;
    double maxAbs = 0.0, maxRel = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        double a = cpu[i], b = gpu[i];
        double absd = fabs(a - b);
        double reld = (a == 0.0) ? absd : absd / fabs(a);
        if (absd > maxAbs) { maxAbs = absd; worst = i; }
        if (reld > maxRel) maxRel = reld;
    }
    std::cerr << "\n[DEBUG] slice " << sliceId
    << "  maxAbs=" << maxAbs
    << "  maxRel=" << maxRel
    << "  worstIdx=" << worst << std::endl;

    if (maxRel > 1e-3) {                // 顯示頭尾 3 個錯最多的元素
        std::cerr << "   CPU=" << cpu[worst]
        << "  GPU=" << gpu[worst] << std::endl;
    }
}


// ---------------- Forward kernels  ----------------
__global__ void forwardKernel(const double* __restrict__ prevF,
                              const double* __restrict__ trans,
                              const double* __restrict__ emis,
                              int prevP, int P,
                              double* __restrict__ curF) {
    extern __shared__ double s[];
    int idx = blockIdx.x;                 // 0..P²-1
    int tid = threadIdx.x;
    int prevSize = prevP * prevP;
    double sum = 0.0;
    for (int j=tid; j<prevSize; j+=blockDim.x)
        sum += prevF[j] * trans[idx*prevSize + j];
    s[tid] = sum; __syncthreads();
    for (int off=blockDim.x/2; off>0; off>>=1){ 
        if(tid<off) s[tid]+=s[tid+off]; __syncthreads(); 
    }
    if (tid==0) 
        curF[idx] = (prevP? s[0] : 1.0) * emis[idx];
}

__global__ void normalizeKernel(double* curF,int size){
    extern __shared__ double s[];
    int tid=threadIdx.x; 
    double v=(tid<size)?curF[tid]:0.0; 
    s[tid]=v; __syncthreads();
    for(int off=blockDim.x/2;off>0;off>>=1){ 
        if(tid<off) s[tid]+=s[tid+off]; __syncthreads(); 
    }
    double total=s[0]; 
    if(total==0.0) 
        total=1.0/size; __syncthreads();
    if(tid<size) 
        curF[tid]/=total;
}

static void gpu_forward(const std::vector<double>& prevF,const std::vector<double>& trans,const std::vector<double>& emis,int prevP,int P,std::vector<double>& outCur){
    int curSize=P*P; 
	int prevSize=prevP*prevP; 
	//const int BLOCK=256; 
	//size_t sh=BLOCK*sizeof(double);
    double *d_prev=nullptr,*d_trans=nullptr,*d_emis=nullptr,*d_cur=nullptr;
    if(prevSize) {
		CUDA_CHECK(cudaMalloc(&d_prev,prevSize*sizeof(double))); 
		CUDA_CHECK(cudaMemcpy(d_prev,prevF.data(),prevSize*sizeof(double),cudaMemcpyHostToDevice));
	}
    if(prevSize){
		CUDA_CHECK(cudaMalloc(&d_trans,curSize*prevSize*sizeof(double))); 
		CUDA_CHECK(cudaMemcpy(d_trans,trans.data(),curSize*prevSize*sizeof(double),cudaMemcpyHostToDevice));
	}
    CUDA_CHECK(cudaMalloc(&d_emis,curSize*sizeof(double))); 
	CUDA_CHECK(cudaMemcpy(d_emis,emis.data(),curSize*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_cur,curSize*sizeof(double)));
	/* ---- launch kernel (一個 block 對一個 state) ---- */
    const int BLOCK = 256;
    size_t shm      = BLOCK * sizeof(double);
    forwardKernel<<<curSize, BLOCK, shm>>>(
        prevP ? d_prev  : nullptr,
        prevP ? d_trans : nullptr,
        d_emis, prevP, P, d_cur);

    /* ---- 將結果搬回 host 再做歸一化 ---- */
    outCur.resize(curSize);
    CUDA_CHECK(cudaMemcpy(outCur.data(), d_cur,
                          curSize * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ---- host-side normalize ---- */
    double sum = std::accumulate(outCur.begin(), outCur.end(), 0.0);
    if (sum == 0.0) 
        sum = 1.0 / curSize;          // underflow fallback
    for (double &v : outCur) 
        v /= sum;

    /* ---- free ---- */
    if (prevSize) { 
        cudaFree(d_prev);  cudaFree(d_trans); 
    }
    cudaFree(d_emis);  
    cudaFree(d_cur);
    // dim3 grid(curSize),block(BLOCK);
    // forwardKernel<<<grid,block,sh>>>(prevP?d_prev:nullptr,prevP?d_trans:nullptr,d_emis,prevP,P,d_cur);
    // normalizeKernel<<<1,256,256*sizeof(double)>>>(d_cur,curSize);
    // outCur.resize(curSize); 
	// CUDA_CHECK(cudaMemcpy(outCur.data(),d_cur,curSize*sizeof(double),cudaMemcpyDeviceToHost));
	// /* ---- CPU 參考值 + 差異檢查 ---- */
	// static int sliceCnt = 0;          // 靜態變數，累積 slice 編號
	// auto cpuRef = cpu_forward_reference(prevF, trans, emis, prevP, P);
	// debug_diff(cpuRef, outCur, P, sliceCnt++);

	// /* ---- transition 索引抽樣 ----
	// * 各 slice 只印第一次 (以免輸出太長)
	// */
	// if (sliceCnt == 1 && prevP) {
	// 	int from = 0, to = 1;                                 // 隨便挑一對 state
	// 	double cpuT = trans[(to*prevP + from) + 0];           // row-major 假設
	// 	std::cerr << "   trans[" << from << "→" << to
	// 			<< "] = " << cpuT << std::endl;
	// }
    // if(prevSize){cudaFree(d_prev); cudaFree(d_trans);} cudaFree(d_emis); cudaFree(d_cur); CUDA_CHECK(cudaDeviceSynchronize());
}

void compute_forward_column_gpu(const std::vector<double>& prevF,const std::vector<double>& trans,const std::vector<double>& emis,int prevP,int P,std::vector<double>& outCur){ 
	gpu_forward(prevF,trans,emis,prevP,P,outCur);
} // wrapper

// ---------------- Backward kernel ----------------
__global__ void backwardKernel(const double* __restrict__ nextB,const double* __restrict__ trans,const double* __restrict__ emisNext,int nextP,int P,double* __restrict__ curB){
    extern __shared__ double s[]; int idx=blockIdx.x; int tid=threadIdx.x; int nxtSize=nextP*nextP;
    double sum=0.0; for(int j=tid;j<nxtSize;j+=blockDim.x){ double tv=trans[j*P*P+idx]; sum+=nextB[j]*tv*emisNext[j]; }
    s[tid]=sum; __syncthreads(); for(int off=blockDim.x/2; off>0; off>>=1){ if(tid<off) s[tid]+=s[tid+off]; __syncthreads(); }
    if(tid==0) curB[idx]=s[0];
}

static void gpu_backward(const std::vector<double>& nextB,const std::vector<double>& trans,const std::vector<double>& emisNext,int nextP,int P,std::vector<double>& outCur){
    int curSize=P*P; 
	int nextSize=nextP*nextP; 
	//const int BLOCK=256; 
	//size_t sh=BLOCK*sizeof(double);
    double *d_next=nullptr,*d_trans=nullptr,*d_emis=nullptr,*d_cur=nullptr;
    if(nextSize){
		CUDA_CHECK(cudaMalloc(&d_next,nextSize*sizeof(double))); 
		CUDA_CHECK(cudaMemcpy(d_next,nextB.data(),nextSize*sizeof(double),cudaMemcpyHostToDevice));
	}
    if(nextSize){
		CUDA_CHECK(cudaMalloc(&d_trans,curSize*nextSize*sizeof(double))); 
		CUDA_CHECK(cudaMemcpy(d_trans,trans.data(),curSize*nextSize*sizeof(double),cudaMemcpyHostToDevice));
	}
    CUDA_CHECK(cudaMalloc(&d_emis,nextSize*sizeof(double))); 
	CUDA_CHECK(cudaMemcpy(d_emis,emisNext.data(),nextSize*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_cur,curSize*sizeof(double)));
	/* ---- launch kernel ---- */
    const int BLOCK = 256;
    size_t shm      = BLOCK * sizeof(double);
    backwardKernel<<<curSize, BLOCK, shm>>>(
        nextP ? d_next  : nullptr,
        nextP ? d_trans : nullptr,
        d_emis, nextP, P, d_cur);

    /* ---- copy back & host normalize ---- */
    outCur.resize(curSize);
    CUDA_CHECK(cudaMemcpy(outCur.data(), d_cur,
                          curSize * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    double sum = std::accumulate(outCur.begin(), outCur.end(), 0.0);
    if (sum == 0.0) sum = 1.0 / curSize;
    for (double &v : outCur) v /= sum;

    /* ---- free ---- */
    if (nextSize) { cudaFree(d_next);  cudaFree(d_trans); }
    cudaFree(d_emis);  cudaFree(d_cur);

    // dim3 grid(curSize),block(BLOCK);
    // backwardKernel<<<grid,block,sh>>>(nextP?d_next:nullptr,nextP?d_trans:nullptr,d_emis,nextP,P,d_cur);
    // normalizeKernel<<<1,256,256*sizeof(double)>>>(d_cur,curSize);
    // outCur.resize(curSize); CUDA_CHECK(cudaMemcpy(outCur.data(),d_cur,curSize*sizeof(double),cudaMemcpyDeviceToHost));
    // if(nxtSize){cudaFree(d_next); cudaFree(d_trans);} cudaFree(d_emis); cudaFree(d_cur); CUDA_CHECK(cudaDeviceSynchronize());
}

void compute_backward_column_gpu(const std::vector<double>& nextB,const std::vector<double>& trans,const std::vector<double>& emisNext,int nextP,int P,std::vector<double>& outCur){ 
	gpu_backward(nextB,trans,emisNext,nextP,P,outCur);
} // wrapper

// ---------------- Viterbi kernel ----------------
__global__ void viterbiKernel(const double* prevV,const double* trans,const double* emisCur,int prevP,int P,double* curV,size_t* backIdx){
    extern __shared__ double buf[]; double* sProb=buf; size_t* sIdx=(size_t*)(buf+blockDim.x);
    int idx=blockIdx.x; int tid=threadIdx.x; int prevSize=prevP*prevP; double best=-1.0; size_t bestj=0;
    for(int j=tid;j<prevSize;j+=blockDim.x){ double val=prevV[j]*trans[idx*prevSize+j]; if(val>best){best=val; bestj=j;} }
    sProb[tid]=best; sIdx[tid]=bestj; __syncthreads();
    for(int off=blockDim.x/2; off>0; off>>=1){ if(tid<off){ if(sProb[tid+off]>sProb[tid]){sProb[tid]=sProb[tid+off]; sIdx[tid]=sIdx[tid+off];} } __syncthreads(); }
    if(tid==0){ curV[idx]=sProb[0]*emisCur[idx]; backIdx[idx]=sIdx[0]; }
}

static void gpu_viterbi(const std::vector<double>& prevV,const std::vector<double>& trans,const std::vector<double>& emisCur,int prevP,int P,std::vector<double>& outCur,std::vector<size_t>& argmax){
    int curSize=P*P; 
	int prevSize=prevP*prevP; 
	//const int BLOCK=256; 
	//size_t sh=BLOCK*(sizeof(double)+sizeof(size_t));
    double *d_prev=nullptr,*d_trans=nullptr,*d_emis=nullptr,*d_cur=nullptr; size_t* d_arg=nullptr;
    if(prevSize){
		CUDA_CHECK(cudaMalloc(&d_prev,prevSize*sizeof(double))); 
		CUDA_CHECK(cudaMemcpy(d_prev,prevV.data(),prevSize*sizeof(double),cudaMemcpyHostToDevice));
	}
    if(prevSize){
		CUDA_CHECK(cudaMalloc(&d_trans,curSize*prevSize*sizeof(double))); 
		CUDA_CHECK(cudaMemcpy(d_trans,trans.data(),curSize*prevSize*sizeof(double),cudaMemcpyHostToDevice));
	}
    CUDA_CHECK(cudaMalloc(&d_emis,curSize*sizeof(double))); 
	CUDA_CHECK(cudaMemcpy(d_emis,emisCur.data(),curSize*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_cur,curSize*sizeof(double))); 
	CUDA_CHECK(cudaMalloc(&d_arg,curSize*sizeof(size_t)));
	/* ---- launch kernel ---- */
    const int BLOCK = 256;
    size_t shm = BLOCK * (sizeof(double) + sizeof(size_t));
    viterbiKernel<<<curSize, BLOCK, shm>>>(
        prevP ? d_prev  : nullptr,
        prevP ? d_trans : nullptr,
        d_emis, prevP, P, d_cur, d_arg);

    /* ---- copy back ---- */
    outCur.resize(curSize);   argmax.resize(curSize);
    CUDA_CHECK(cudaMemcpy(outCur.data(), d_cur,
                          curSize * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(argmax.data(), d_arg,
                          curSize * sizeof(size_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ---- host normalize (for consistency) ---- */
    double sum = std::accumulate(outCur.begin(), outCur.end(), 0.0);
    if (sum == 0.0) sum = 1.0 / curSize;
    for (double &v : outCur) v /= sum;

    /* ---- free ---- */
    if (prevSize) { cudaFree(d_prev);  cudaFree(d_trans); }
    cudaFree(d_emis);  cudaFree(d_cur);  cudaFree(d_arg);
    // dim3 grid(curSize),block(BLOCK);
    // viterbiKernel<<<grid,block,sh>>>(prevP?d_prev:nullptr,prevP?d_trans:nullptr,d_emis,prevP,P,d_cur,d_arg);
    // normalizeKernel<<<1,256,256*sizeof(double)>>>(d_cur,curSize);
    // outCur.resize(curSize); argmax.resize(curSize);
    // CUDA_CHECK(cudaMemcpy(outCur.data(),d_cur,curSize*sizeof(double),cudaMemcpyDeviceToHost));
    // CUDA_CHECK(cudaMemcpy(argmax.data(),d_arg,curSize*sizeof(size_t),cudaMemcpyDeviceToHost));
    // if(prevSize){cudaFree(d_prev); cudaFree(d_trans);} cudaFree(d_emis); cudaFree(d_cur); cudaFree(d_arg); CUDA_CHECK(cudaDeviceSynchronize());
}

void compute_viterbi_column_gpu(const std::vector<double>& prevV,const std::vector<double>& trans,const std::vector<double>& emisCur,int prevP,int P,std::vector<double>& outCur,std::vector<size_t>& argmax){ 
	gpu_viterbi(prevV,trans,emisCur,prevP,P,outCur,argmax);
} // wrapper