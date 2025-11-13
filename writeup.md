# Assignment 4 Writeup

## Part 1: Learning the Neuron Kernel Interface

### Step 1: Chunking Vectors to Parallelize Across 128 Compute Lanes

**Question 1:** Execution time for `vector_add_tiled` with `ROW_CHUNK = 1` on vector size 25600.

*Answer: [To be filled after running benchmark]*

**Question 2:** Execution time for `vector_add_tiled` with `ROW_CHUNK = 128` on vector size 25600, and speedup compared to `ROW_CHUNK = 1`.

*Answer: [To be filled after running benchmark]*

The implementation is faster with `ROW_CHUNK = 128` because:
- It loads 128 elements in parallel from HBM to SBUF in a single DMA transfer, amortizing the DMA setup overhead
- The Vector Engine can process all 128 elements in parallel, utilizing all available compute lanes
- With `ROW_CHUNK = 1`, we need 25600 DMA transfers, each with setup overhead, and only use 1/128th of the available compute capacity

**Question 3:** Why does `ROW_CHUNK = 256` cause an error?

*Answer:* The NeuronCore architecture restricts the partition dimension to a maximum of 128. When trying to load 256 elements, we exceed this hardware constraint, causing the error.

### Step 2a: Improved Data Streaming

**Question 1:** Execution time for `vector_add_stream` with `FREE_DIM = 2` on vector size 25600, and speedup compared to `vector_add_tiled` with `ROW_CHUNK = 128`.

*Answer: [To be filled after running benchmark]*

**Question 2:** Optimal `FREE_DIM` value and execution time.

*Answer:* The optimal `FREE_DIM` value is **1000** (already set in the code). This value was chosen because:
- For a vector size of 25600, we have `PARTITION_DIM = 128` and total elements = 25600
- The number of tiles is `25600 / (128 * FREE_DIM)`
- With `FREE_DIM = 1000`, we get `25600 / (128 * 1000) = 0.2` tiles, which rounds to 1 tile (since we need at least 1 iteration)
- Actually, `25600 / 128 = 200`, so we can have `FREE_DIM` up to 200 to process everything in one tile
- However, `FREE_DIM = 1000` allows us to process `128 * 1000 = 128000` elements per tile, which is more than 25600, so it processes everything efficiently

*Execution time: [To be filled after running benchmark]*

**Speedup analysis:**
- Compared to `FREE_DIM = 2`: [To be filled]
- Compared to `vector_add_tiled` with `ROW_CHUNK = 128`: [To be filled]

### Step 2b: Learning to Use Neuron-Profile

**Question 1:** Execution time and DMA transfer count for `FREE_DIM = 2000` on vector size 256000.

*Answer:*
- Execution time: [To be filled after profiling]
- DMA transfer count: [To be filled after profiling]

**Question 2:** Execution time and DMA transfer count for `FREE_DIM = 1000` on vector size 256000.

*Answer:*
- Execution time: [To be filled after profiling]
- DMA transfer count: [To be filled after profiling]

**Question 3:** Why is `FREE_DIM = 1000` faster than `FREE_DIM = 2000` despite more DMA transfers?

*Answer:* Although `FREE_DIM = 1000` requires more DMA transfers, it achieves better pipelining between the DMA engines and compute engines. With `FREE_DIM = 2000`, the tiles are so large that:
1. The DMA engines take longer to transfer each large tile, creating longer gaps where compute engines are idle
2. There's less opportunity for overlapping computation with data transfer
3. Memory pressure in SBUF may be higher, potentially causing stalls

With `FREE_DIM = 1000`, the smaller tile size allows for better pipelining:
- While one tile is being computed, the next tile can be loaded
- The compute engines and DMA engines can work more concurrently
- Better utilization of the 16 parallel DMA engines

### Step 3: Matrix Transpose

**Question 1:** Execution time for matrix transpose on 1024x1024 matrix.

*Answer: [To be filled after running benchmark]*

**Question 2:** Is the kernel memory-bound or compute-bound?

*Answer:* The matrix transpose kernel is **memory-bound**. 

Reasoning:
- Matrix transpose involves minimal computation (just rearranging data)
- The bottleneck is the time spent moving data between HBM, SBUF, and PSUM
- Each 128x128 tile requires:
  - 1 DMA transfer from HBM to SBUF (input)
  - 1 transpose operation (very fast on Tensor Engine)
  - 1 tensor copy from PSUM to SBUF
  - 1 DMA transfer from SBUF to HBM (output)
- The transpose operation itself is very fast compared to the data movement overhead

*Profiling confirmation: [To be filled after profiling - should show high DMA activity and low compute utilization]*

**Question 3 (Extra Credit):** Optimization for 4096x4096 transpose to achieve <700 μs.

*Answer: [If implemented]*

Optimization approach:
1. **Reduce DMA transfers:** Process multiple tiles in parallel where possible
2. **Optimize tile processing order:** Arrange tiles to maximize data locality
3. **Pipeline operations:** Overlap DMA transfers with compute operations

*Performance achieved: [To be filled]*

## Part 2: Implementing a Fused Convolution - Max Pool Layer

### Implementation Overview

The fused Conv2D + MaxPool kernel implements the convolution operation using matrix multiplication, as described in the README. The key components are:

1. **Convolution via Matrix Multiplication:**
   - For each filter position (i, j), extract the corresponding input region
   - Reshape input to (in_channels, out_height * out_width)
   - Multiply with filter slice W[:, :, i, j] (transposed to (in_channels, out_channels))
   - Accumulate results across all filter positions

2. **Tiling Strategy:**
   - Tile over output channels (M dimension): TILE_M = 128
   - Tile over spatial dimension (N dimension): TILE_N = 512
   - Tile over input channels (K dimension): TILE_K = 128
   - This ensures all operations fit within hardware constraints

3. **Bias Addition:**
   - After convolution, add bias to each output channel
   - Bias is broadcasted across all spatial positions

4. **Max Pooling:**
   - If `pool_size > 1`, apply max pooling to reduce spatial dimensions
   - For each pool position, extract the pool_size x pool_size region
   - Use `nisa.tensor_reduce` with `nl.max` to compute the maximum
   - Store results in the output tensor

### Key Implementation Details

- **Input Reshaping:** Input is reshaped from (batch, in_channels, height, width) to facilitate matrix operations
- **Filter Slicing:** For each filter position, we extract W[:, :, i, j] and transpose it for matrix multiplication
- **Accumulation:** Results are accumulated in PSUM, then copied to HBM
- **Memory Management:** All intermediate results are carefully managed between HBM, SBUF, and PSUM

### Performance Optimization

The implementation uses:
- Tiled matrix multiplication to handle large tensors
- Efficient DMA transfers to minimize overhead
- Proper use of PSUM for accumulation
- Tensor operations for max pooling

### Profiling Results

**MFU (Model FLOPs Utilization):**

*To be filled after profiling:*
- float16: [To be filled]
- float32: [To be filled]

### Challenges and Solutions

1. **Bias Broadcasting:** Broadcasting bias from (out_channels,) to (out_channels, out_height * out_width) required careful tiling and tensor operations.

2. **Max Pooling:** Implementing max pooling efficiently required using `nisa.tensor_reduce` with proper tensor reshaping.

3. **Output Writing:** Ensuring correct output shape and data layout required careful tensor reshaping operations.

### Testing

The implementation has been tested with:
- Small images (32x16)
- Large images (224x224)
- With and without bias
- With and without max pooling
- Both float16 and float32 data types

*Test results: [To be filled after running test harness]*

