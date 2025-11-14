import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""
@nki.compiler.skip_middle_end_transformations
@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size

    tile_in = int(nl.tile_size.pmax)
    tile_out = int(nl.tile_size.gemm_stationary_fmax)
    tile_free_max = int(nl.tile_size.gemm_moving_fmax)

    assert in_channels % tile_in == 0, "Input channels must be multiples of 128"
    assert out_channels % tile_out == 0, "Output channels must be multiples of 128"
    assert tile_free_max >= out_width, "Output width must fit in tensor engine free dim"
    if pool_size > 1:
        assert out_height % pool_size == 0 and out_width % pool_size == 0, "Pooling requires divisible dims"

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    compute_dtype = nl.float32 if X.dtype == nl.float16 else X.dtype
    output_dtype = X.dtype

    max_rows = max(1, tile_free_max // out_width)
    if pool_size > max_rows:
        raise AssertionError("Free dimension too small for requested pool size")
    max_rows = min(max_rows, out_height)
    assert out_height >= pool_size, "Pool size exceeds output height"
    max_rows -= max_rows % pool_size
    if max_rows == 0:
        max_rows = pool_size

    rows_per_block = max_rows
    while rows_per_block > pool_size and out_height % rows_per_block != 0:
        rows_per_block -= pool_size
    if out_height % rows_per_block != 0:
        rows_per_block = pool_size
    assert rows_per_block > 0
    assert out_height % rows_per_block == 0
    assert rows_per_block * out_width <= tile_free_max

    rows = rows_per_block
    rows_out = rows // pool_size
    cols_out = out_width // pool_size if pool_size > 1 else out_width
    n_cols = rows * out_width
    assert n_cols <= tile_free_max

    num_row_tiles = out_height // rows
    num_out_tiles = out_channels // tile_out
    num_in_tiles = in_channels // tile_in

    for b in nl.affine_range(batch_size):
        for oc_tile_idx in nl.affine_range(num_out_tiles):
            oc_start = oc_tile_idx * tile_out

            bias_tile = nl.ndarray((tile_out, 1), dtype=bias.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                src=bias[oc_start : oc_start + tile_out],
                dst=bias_tile[:, 0],
            )
            if bias_tile.dtype != compute_dtype:
                bias_tile = nl.copy(bias_tile, dtype=compute_dtype)

            for h_tile_idx in nl.affine_range(num_row_tiles):
                row_start = h_tile_idx * rows_per_block

                res_psum = nl.zeros((tile_out, n_cols), dtype=compute_dtype, buffer=nl.psum)

                expanded_rows = rows + filter_height - 1
                expanded_cols = out_width + filter_width - 1

                for ic_tile_idx in nl.affine_range(num_in_tiles):
                    ic_start = ic_tile_idx * tile_in

                    input_block = nl.ndarray(
                        (tile_in, expanded_rows, expanded_cols), dtype=X.dtype, buffer=nl.sbuf
                    )
                    nisa.dma_copy(
                        src=X[
                            b,
                            ic_start : ic_start + tile_in,
                            row_start : row_start + expanded_rows,
                            0 : expanded_cols,
                        ],
                        dst=input_block,
                    )

                    weights_block = nl.ndarray(
                        (tile_out, tile_in, filter_height, filter_width),
                        dtype=W.dtype,
                        buffer=nl.sbuf,
                    )
                    nisa.dma_copy(
                        src=W[
                            oc_start : oc_start + tile_out,
                            ic_start : ic_start + tile_in,
                            0:filter_height,
                            0:filter_width,
                        ],
                        dst=weights_block,
                    )

                    for fh in range(filter_height):
                        for fw in range(filter_width):
                            weight_tile = nisa.tensor_copy(weights_block[:, :, fh, fw])
                            weight_psum = nisa.nc_transpose(weight_tile)
                            lhs_tile = nisa.tensor_copy(weight_psum)
                            if lhs_tile.dtype != compute_dtype:
                                lhs_tile = nl.copy(lhs_tile, dtype=compute_dtype)

                            input_tile = nisa.tensor_copy(
                                input_block[:, fh : fh + rows, fw : fw + out_width]
                            )
                            rhs_tile = input_tile.reshape((tile_in, n_cols))
                            if rhs_tile.dtype != compute_dtype:
                                rhs_tile = nl.copy(rhs_tile, dtype=compute_dtype)

                            res_psum += nisa.nc_matmul(lhs_tile, rhs_tile)

                res_tile = nisa.tensor_copy(res_psum)
                res_tile = res_tile.reshape((tile_out, n_cols))
                res_with_bias = nisa.tensor_scalar(res_tile, nl.add, bias_tile)
                res_with_bias = res_with_bias.reshape((tile_out, rows, out_width))

                if pool_size > 1:
                    width_grouped = res_with_bias.reshape(
                        (tile_out, rows, cols_out, pool_size)
                    )
                    reduced_w = nisa.tensor_reduce(nl.max, width_grouped, axis=[3])
                    height_grouped = reduced_w.reshape(
                        (tile_out, rows_out, pool_size, cols_out)
                    )
                    pooled_tile = nisa.tensor_reduce(nl.max, height_grouped, axis=[2])
                else:
                    pooled_tile = res_with_bias

                if pooled_tile.dtype != output_dtype:
                    pooled_tile = nl.copy(pooled_tile, dtype=output_dtype)

                row_out_start = row_start // pool_size
                nisa.dma_copy(
                    src=pooled_tile,
                    dst=X_out[
                        b,
                        oc_start : oc_start + tile_out,
                        row_out_start : row_out_start + rows_out,
                        0:cols_out,
                    ],
                )

    return X_out
