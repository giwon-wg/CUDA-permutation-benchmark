#include "CudaInterface.h"
#include <string>
#include <vector>
#include <iostream>

extern "C" {
    JNIEXPORT jint JNICALL Java_CudaInterface_checkPermutation(
        JNIEnv *env, jobject obj, jstring jseq, jstring jperm
    );

    JNIEXPORT jint JNICALL Java_CudaInterface_checkPermutationByte(
        JNIEnv *env, jobject obj, jbyteArray jseqArr, jbyteArray jpermArr
    );

    JNIEXPORT jintArray JNICALL Java_CudaInterface_checkPermutationBatch(
            JNIEnv* env, jobject obj, jbyteArray jsequence, jobjectArray jpermArray
        );
}

// 문자열 방식
JNIEXPORT jint JNICALL Java_CudaInterface_checkPermutation(
    JNIEnv *env, jobject obj, jstring jseq, jstring jperm
) {
    jsize seqCharLen = env->GetStringLength(jseq);
    jsize permCharLen = env->GetStringLength(jperm);

    int maxBytes = 256;
    char* seqBuf = new char[maxBytes];
    char* permBuf = new char[maxBytes];

    env->GetStringUTFRegion(jseq, 0, seqCharLen, seqBuf);
    env->GetStringUTFRegion(jperm, 0, permCharLen, permBuf);

    seqBuf[seqCharLen] = '\0';
    permBuf[permCharLen] = '\0';

    std::string seqStr(seqBuf);
    std::string permStr(permBuf);

    std::cout << "[DLL] sequence: " << seqStr << std::endl;
    std::cout << "[DLL] perm    : " << permStr << std::endl;

    size_t index = seqStr.find(permStr);
    std::cout << "[DLL] find() index: " << index << std::endl;
    std::cout << "[DLL] contains? " << (index != std::string::npos ? "YES" : "NO") << std::endl;

    delete[] seqBuf;
    delete[] permBuf;

    return (index != std::string::npos) ? 1 : 0;
}

// 바이트 배열
JNIEXPORT jint JNICALL Java_CudaInterface_checkPermutationByte(
    JNIEnv *env, jobject obj, jbyteArray jseqArr, jbyteArray jpermArr
) {
    jsize seqLen = env->GetArrayLength(jseqArr);
    jsize permLen = env->GetArrayLength(jpermArr);

    jbyte* seq = env->GetByteArrayElements(jseqArr, nullptr);
    jbyte* perm = env->GetByteArrayElements(jpermArr, nullptr);

    std::string seqStr(reinterpret_cast<char*>(seq), seqLen);
    std::string permStr(reinterpret_cast<char*>(perm), permLen);

    size_t index = seqStr.find(permStr);

    env->ReleaseByteArrayElements(jseqArr, seq, 0);
    env->ReleaseByteArrayElements(jpermArr, perm, 0);

    return (index != std::string::npos) ? 1 : 0;
}

__global__ void checkPermutationsKernel(
    const char* sequence, int seqLen,
    const char* permutations, int permLen, int totalPerms,
    int* results
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalPerms) return;

    const char* perm = permutations + idx * permLen;
    bool matchFound = false;

    for (int i = 0; i <= seqLen - permLen; i++) {
        bool matched = true;
        for (int j = 0; j < permLen; j++) {
            if (sequence[i + j] != perm[j]) {
                matched = false;
                break;
            }
        }
        if (matched) {
            matchFound = true;
            break;
        }
    }

    results[idx] = matchFound ? 1 : 0;
}

// CUDA 병렬 검사 (배치)
JNIEXPORT jintArray JNICALL Java_CudaInterface_checkPermutationBatch(
    JNIEnv* env, jobject obj,
    jbyteArray jsequence,
    jobjectArray jpermArray
) {
    jsize seqLen = env->GetArrayLength(jsequence);
    jbyte* h_seq = env->GetByteArrayElements(jsequence, 0);

    jsize numPerms = env->GetArrayLength(jpermArray);
    jbyteArray firstPerm = (jbyteArray) env->GetObjectArrayElement(jpermArray, 0);
    jsize permLen = env->GetArrayLength(firstPerm);

    std::vector<jbyte> h_perms(numPerms * permLen);
    for (int i = 0; i < numPerms; i++) {
        jbyteArray jperm = (jbyteArray) env->GetObjectArrayElement(jpermArray, i);
        env->GetByteArrayRegion(jperm, 0, permLen, &h_perms[i * permLen]);
    }

    char *d_seq, *d_perms;
    int *d_result, *h_result = new int[numPerms];
    cudaMalloc(&d_seq, seqLen);
    cudaMalloc(&d_perms, numPerms * permLen);
    cudaMalloc(&d_result, numPerms * sizeof(int));

    cudaMemcpy(d_seq, h_seq, seqLen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_perms, h_perms.data(), numPerms * permLen, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (numPerms + threadsPerBlock - 1) / threadsPerBlock;
    checkPermutationsKernel<<<blocks, threadsPerBlock>>>(
        d_seq, seqLen,
        d_perms, permLen, numPerms,
        d_result
    );
    cudaDeviceSynchronize();

    cudaMemcpy(h_result, d_result, numPerms * sizeof(int), cudaMemcpyDeviceToHost);

    jintArray jResult = env->NewIntArray(numPerms);
    env->SetIntArrayRegion(jResult, 0, numPerms, (const jint*) h_result);

    cudaFree(d_seq);
    cudaFree(d_perms);
    cudaFree(d_result);
    env->ReleaseByteArrayElements(jsequence, h_seq, 0);
    delete[] h_result;

    return jResult;
}
