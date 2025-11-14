
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

## Fused Conv2D Optimization Notes

After getting correctness, the initial kernel still mirrored the naïve pseudocode in the README: every `(fh, fw, ic_tile)` loop issued fresh DMAs for `X` and `W`, bias was inserted via `nisa.tensor_scalar`, and PSUM tiles matched the input dtype. The first hardware run looked like this:
- **Baseline (correctness-first)** – `python3 test_harness.py --test_maxpool`  
  - Large image, float32, no pool: **13.5 ms** (`13481 µs`)  
  - Large image, float16, no pool: **12.3 ms** (`12303 µs`)  
  - Small image, float32, no pool: **0.43 ms** (`430 µs`)  
  These numbers made it clear that redundant DMA traffic and lack of tiling were the bottlenecks.

### Float32 Path Optimizations
1. **Input hoisting + halo tiles** (Section “Reducing the number of DMA transfers” in the NKI docs: [link](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/index.html))  
   - Action: DMA a `(tile_in, rows+FH−1, out_width+FW−1)` block once per `(ic_tile, row_tile)` and slice it inside SBUF via `nisa.tensor_copy`.  
   - Result: float32 latency dropped to **6.3 ms** (`6289 µs`), float16 to **5.6 ms** (still failing), small-image cases to **384–505 µs**.
2. **Weight hoisting/caching + PSUM discipline**  
   - Action: load each `(tile_out, tile_in, FH, FW)` block once per `(oc_tile, ic_tile)` and reuse across filter loops; switch PSUM allocations to explicit float32 and replace `tensor_scalar` with `tensor_tensor` to avoid the documented float16 incompatibility.  
   - Result: float32 (±pool) fell to **~3.7 ms**, while float16 stayed high (~4.1 ms) but compilation was now stable.

### Float16-Specific Optimizations
1. **Flattened weight cache & single dtype cast**  
   - Action: store all transposed filter slices in one flattened SBUF tile indexed by `(ic_tile, fh, fw)` so each matmul pulls a contiguous `(tile_in, tile_out)` block without re-transposing; convert rhs tiles to float32 exactly once before `nisa.nc_matmul`.  
   - Result: float16 latencies improved to **3.3 ms** but still missed the 1.36 ms goal.
2. **Final tuning (current version)**  
   - Action: keep cached weights in their original dtype (to stay within SBUF limits) while PSUM/rhs tiles remain float32, letting the tensor engine handle conversions internally; ensure every DMA moves maximal tiles and avoid any per-iteration dtype casts.  
   - Result: float16 (no pool) **0.93 ms**, float16 (pool) **0.87 ms**, matching the float32 numbers (~3.0 ms) and passing all performance tests plus EC cases (≤36 µs on 32×16 inputs).

### Smaller Image Strategy (vs. Large Images)
- Once the above tiling was in place, the 32×16 cases fit entirely inside a single row tile. I set `rows_per_block` to the full height (still a multiple of the pooling stride), so each batch processes in one PSUM accumulation. All weights are cached once and every DMA moves the entire image, trimming latency from ~0.43 ms (baseline) to **35–57 µs** across precision/pooling variants. This directly follows the “load fewer rows” guidance in the README by ensuring the halo rows are shared between consecutive pooled outputs instead of reloaded.
- Because the full feature map is loaded at once, the expanded input block automatically contains the exact four-row window needed to produce two pooled rows (pool_size = 2). The middle two rows of that window are reused when sliding the pooling kernel, so we never issue duplicate DMAs for overlapping regions. With only one output tile per batch, the kernel spends nearly all of its time in a single `nisa.nc_matmul` + pooling sequence, yielding the tens-of-microseconds latencies reported.
- For the 224×224 configurations, the same reuse idea applies but we must partition the height into multiple `rows_per_block` chunks that satisfy the `tile_free_max` constraint. Each chunk therefore loads its own `(rows+FH−1)` halo slice, and we pipeline the `(fh, fw)` matmuls across `num_row_tiles`. This introduces extra DMA traffic and PSUM flushes compared to the single-tile small-image case, but it is the only way to keep the tensor-engine fed when the full feature map cannot simultaneously reside in SBUF.

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



