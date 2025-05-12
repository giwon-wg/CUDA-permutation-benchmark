###  **TIL - 2025.04.15**

## **오늘 배운 것 개요**
- **[CUDA + JNI 연동 방식 학습](https://github.com/giwon-wg/CUDA-permutation-benchmark)**
- **Java에서 DLL 호출 후 GPU 병렬 연산 수행**
- **3백만 개 이상의 순열을 GPU와 CPU 방식으로 비교**
- **성능 차이 실험을 통해 CUDA의 이점을 확인**


---

## **1. 이 실험을 시작한 이유**

이번 실험은 애니메이션 "스즈미야 하루히의 우울"에서 착안한 **Haruhi Problem**에서 시작됐다.  
이 문제는 "12개의 에피소드를 가능한 모든 순열로 시청하려면, 각 순열이 포함된 최소한의 시청 길이는 얼마인가?"라는 흥미로운 질문에서 출발한다.

처음에는 CPU 방식으로 순열을 생성하고 문자열에 포함되는지를 검사하는 프로그램을 만들었는데,  
**n=10 (362만 개 순열)**까지는 괜찮았지만  
**n=11~12부터는 연산 시간이 폭증**한다는 문제에 부딪혔다.

그래서 이 문제를 **병렬화하여 해결할 수 있을까?** → CUDA를 활용하는 방향으로 확장하게 됐다.

---

## **2. 트러블 슈팅**

1. **JNI 연동과 DLL 빌드 에러**
   - `cl.exe` 경로 인식 안됨 → Visual Studio 커맨드 프롬프트 해결
   - `nvcc` 경로 문제 해결 → 환경 변수 및 경로 확인

2. **헤더 파일 연동 오류**
   - `javac -h .` 명령 누락으로 `CudaInterface.h`가 생성되지 않아 JNI 인식 실패

3. **인코딩 문제**
   - 코드 페이지 949에서 CUDA 빌드시 한글 포함된 주석이 깨짐 → UTF-8 저장 필요

4. **std::vector 인식 실패**
   - CUDA `.cu` 파일은 `#include <vector>` 명시 필요 → 오류 해결

5. **`jint` vs `int*` 타입 충돌**
   - JNI에서 `SetIntArrayRegion()` 호출 시 명시적 캐스팅 필요

6. **CUDA에서 바이트 배열 비교 로직**
   - `std::string::find()`는 CUDA에서 사용 불가 → `memcmp` 대신 루프 기반 문자열 매칭 구현

---

## **3. 실험 목표**

### **1. 순열 포함 검사**
- 문자열 안에 특정 순열이 포함되어 있는지 검사하는 문제 (Haruhi Problem 응용)
- `sequence` 문자열과 `perm` 순열을 비교해 포함 여부를 판단

### **2. 성능 비교**
- CPU 방식: Java → JNI → C++ → 순차적 검사
- CUDA 방식: Java → JNI → CUDA 커널 → 병렬 검사

---

## **4. 코드 구성**

### **1. Java 측 Main 구조**
- 전체 순열 생성 (10! = 3,628,800개)
- CPU 방식 vs CUDA 방식 검사 시간 측정

#### **코드 예시**
```java
Instant startCpu = Instant.now();
int countCpu = 0;
for (byte[] perm : permBytes) {
    int result = cuda.checkPermutationByte(sequenceBytes, perm);
    if (result == 1) countCpu++;
}
Instant endCpu = Instant.now();

Instant startCuda = Instant.now();
int[] resultArray = cuda.checkPermutationBatch(sequenceBytes, permBytes);
int countCuda = 0;
for (int r : resultArray) {
    if (r == 1) countCuda++;
}
Instant endCuda = Instant.now();
```

---

### **2. JNI + CUDA 커널 구조**

#### JNI 함수 구현
```cpp
JNIEXPORT jintArray JNICALL Java_CudaInterface_checkPermutationBatch(...) {
    // Java 배열 → C 배열로 복사
    // CUDA malloc/copy
    // <<<kernel>>> 실행
    // 결과 → Java int[] 반환
}
```

#### CUDA 커널
```cpp
__global__ void checkPermutationsKernel(...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    ...
    results[idx] = (matchFound) ? 1 : 0;
}
```

---

## **5. 실험 결과**

| 항목 | 결과 |
|------|------|
| 순열 수 | 3,628,800개  
| 포함된 순열 수 | 10개 (회전형 시퀀스 기준)  
| CPU 검사 시간 | **1126ms**  
| CUDA 검사 시간 | **321ms**  
| 속도 향상률 | **약 3.5배** 빠름

---
