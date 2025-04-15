import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

public class Main {
	public static void main(String[] args) {
		CudaInterface cuda = new CudaInterface();


		// 순열 생성
		String val = "0123456789"; // n = 10
		List<String> perms = new ArrayList<>();
		permute("", val, perms);
		System.out.println("총 순열 수: " + perms.size());


		// 시퀀스 생성 및 바이트 변환
		String sequence = "01234567890123456789";
		byte[] sequenceBytes = sequence.getBytes(StandardCharsets.US_ASCII);
		byte[][] permBytes = new byte[perms.size()][val.length()];
		for (int i = 0; i < perms.size(); i++) {
			permBytes[i] = perms.get(i).getBytes(StandardCharsets.US_ASCII);
		}

		// CPU 방식 검사
		int countCpu = 0;
		Instant startCpu = Instant.now();
		for (byte[] perm : permBytes) {
			int result = cuda.checkPermutationByte(sequenceBytes, perm);
			if (result == 1) {
				countCpu++;
			}
		}
		Instant endCpu = Instant.now();

		// CUDA 방식 검사
		int countCuda = 0;
		Instant startCuda = Instant.now();
		int[] cudaResults = cuda.checkPermutationBatch(sequenceBytes, permBytes);
		for (int i = 0; i < cudaResults.length; i++) {
			if (cudaResults[i] == 1) {
				countCuda++;
			}
		}
		Instant endCuda = Instant.now();

		// 결과 비교 출력
		System.out.println("CPU 방식 포함된 순열 수: " + countCpu);
		System.out.println("CPU 검사 시간(ms): " + Duration.between(startCpu, endCpu).toMillis());

		System.out.println("CUDA 방식 포함된 순열 수: " + countCuda);
		System.out.println("CUDA 검사 시간(ms): " + Duration.between(startCuda, endCuda).toMillis());

		System.out.println("성능 차이: " + (Duration.between(startCpu, endCpu).toMillis() - Duration.between(startCuda, endCuda).toMillis()));
	}

	// 순열 생성
	private static void permute(String prefix, String remaining, List<String> result) {
		if (remaining.isEmpty()) {
			result.add(prefix);
			return;
		}
		for (int i = 0; i < remaining.length(); i++) {
			permute(
				prefix + remaining.charAt(i),
				remaining.substring(0, i) + remaining.substring(i + 1),
				result
			);
		}
	}
}
