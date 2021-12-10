# Model optimization using different convolutions
### Model: 
- EDSR (Single Image Super Resolution)
### Convs: 
- Standard
- Depthwise separable

<br />

# Test 

## EDSR standard convolution residual blocks
```bash
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
void gemv2N_kernel<int, int, float2, float2, float2,...         0.00%       0.000us         0.00%       0.000us       0.000us     236.753ms        22.28%     236.753ms      15.260us         15515  
ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile14...         0.00%       0.000us         0.00%       0.000us       0.000us     184.237ms        17.33%     184.237ms       5.118ms            36  
void fft2d_r2c_32x32<float, false, 0u, false>(float2...         0.00%       0.000us         0.00%       0.000us       0.000us     137.056ms        12.89%     137.056ms       8.834us         15515  
void fft2d_c2r_32x32<float, false, false, 0u, false,...         0.00%       0.000us         0.00%       0.000us       0.000us     115.993ms        10.91%     115.993ms       7.476us         15515  
void implicit_convolve_sgemm<float, float, 128, 5, 5...         0.00%       0.000us         0.00%       0.000us       0.000us      85.559ms         8.05%      85.559ms      12.223ms             7  
void implicit_convolve_sgemm<float, float, 512, 6, 8...         0.00%       0.000us         0.00%       0.000us       0.000us      75.206ms         7.08%      75.206ms      37.603ms             2  
void precomputed_convolve_sgemm<float, 512, 6, 8, 3,...         0.00%       0.000us         0.00%       0.000us       0.000us      36.294ms         3.41%      36.294ms      36.294ms             1  
void precomputed_convolve_sgemm<float, 128, 5, 5, 3,...         0.00%       0.000us         0.00%       0.000us       0.000us      35.486ms         3.34%      35.486ms      11.829ms             3  
void explicit_convolve_sgemm<float, int, 128, 6, 7, ...         0.00%       0.000us         0.00%       0.000us       0.000us      34.895ms         3.28%      34.895ms      34.895ms             1  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      28.460ms         2.68%      28.460ms     837.059us            34  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 4.228s
Self CUDA time total: 1.063s

Total number of params : 1332931
```

<br />

## EDSR depthwise separable convolution residual blocks
```bash
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
void gemv2N_kernel<int, int, float2, float2, float2,...         0.00%       0.000us         0.00%       0.000us       0.000us     209.520ms        23.93%     209.520ms      16.301us         12853  
void fft2d_r2c_32x32<float, false, 0u, false>(float2...         0.00%       0.000us         0.00%       0.000us       0.000us     111.454ms        12.73%     111.454ms       8.671us         12854  
void fft2d_c2r_32x32<float, false, false, 0u, false,...         0.00%       0.000us         0.00%       0.000us       0.000us      88.412ms        10.10%      88.412ms       6.879us         12853  
void implicit_convolve_sgemm<float, float, 512, 6, 8...         0.00%       0.000us         0.00%       0.000us       0.000us      75.168ms         8.59%      75.168ms      37.584ms             2  
void spatialDepthwiseConvolutionUpdateOutput<float, ...         0.00%       0.000us         0.00%       0.000us       0.000us      73.835ms         8.43%      73.835ms       2.307ms            32  
void implicit_convolve_sgemm<float, float, 1024, 5, ...         0.00%       0.000us         0.00%       0.000us       0.000us      62.712ms         7.16%      62.712ms       1.844ms            34  
void implicit_convolve_sgemm<float, float, 128, 5, 5...         0.00%       0.000us         0.00%       0.000us       0.000us      53.586ms         6.12%      53.586ms      10.717ms             5  
ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile14...         0.00%       0.000us         0.00%       0.000us       0.000us      36.876ms         4.21%      36.876ms      12.292ms             3  
void precomputed_convolve_sgemm<float, 512, 6, 8, 3,...         0.00%       0.000us         0.00%       0.000us       0.000us      36.271ms         4.14%      36.271ms      36.271ms             1  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      28.678ms         3.28%      28.678ms     843.471us            34  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 4.060s
Self CUDA time total: 875.515ms

Total number of params : 304835
```
