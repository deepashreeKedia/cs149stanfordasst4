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

