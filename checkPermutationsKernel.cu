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
