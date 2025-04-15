public class CudaInterface {
	public native int checkPermutation(String sequence, String permutation);

	public native int checkPermutationByte(byte[] sequence, byte[] permutation);

	public native int[] checkPermutationBatch(byte[] sequence, byte[][] permutations);

	static {
		System.loadLibrary("cudalib");
	}
}
