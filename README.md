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
                                        model_inference         0.07%       2.997ms       100.00%        4.324s        4.324s       0.000us         0.00%        1.040s        1.040s             1  
                                           aten::conv2d         0.02%       1.008ms        99.58%        4.306s     123.028ms       0.000us         0.00%     976.043ms      27.887ms            35  
                                      aten::convolution         0.01%     413.000us        99.55%        4.305s     122.999ms       0.000us         0.00%     976.043ms      27.887ms            35  
                                     aten::_convolution         0.02%       1.007ms        99.54%        4.305s     122.987ms       0.000us         0.00%     976.043ms      27.887ms            35  
                                aten::cudnn_convolution         3.90%     168.809ms        99.47%        4.301s     122.892ms     947.580ms        91.12%     947.580ms      27.074ms            35  
                                        cudaMemsetAsync        42.40%        1.834s        42.40%        1.834s     458.405ms       0.000us         0.00%       0.000us       0.000us             4  
                                               cudaFree        32.43%        1.402s        32.43%        1.402s     233.700ms       0.000us         0.00%       0.000us       0.000us             6  
                                   cudaEventSynchronize        13.54%     585.471ms        13.54%     585.471ms      27.880ms       0.000us         0.00%       0.000us       0.000us            21  
                                       cudaLaunchKernel         6.22%     269.069ms         6.22%     269.069ms       6.136us       0.000us         0.00%       0.000us       0.000us         43849  
                                             cudaMalloc         1.13%      48.710ms         1.13%      48.710ms       1.316ms       0.000us         0.00%       0.000us       0.000us            37  
                                              aten::mul         0.03%       1.086ms         0.31%      13.583ms     424.469us      24.587ms         2.36%      24.587ms     768.344us            32  
                                            aten::empty         0.02%     720.000us         0.29%      12.570ms     179.571us       0.000us         0.00%       0.000us       0.000us            70  
                                             aten::add_         0.03%       1.446ms         0.04%       1.875ms      36.058us      48.415ms         4.66%      48.415ms     931.058us            52  
                                    cudaStreamWaitEvent         0.04%       1.813ms         0.04%       1.813ms       0.124us       0.000us         0.00%       0.000us       0.000us         14633  
                                          cudaHostAlloc         0.03%       1.155ms         0.03%       1.155ms       1.155ms       0.000us         0.00%       0.000us       0.000us             1  
                                          aten::reshape         0.01%     326.000us         0.03%       1.134ms      30.649us       0.000us         0.00%       3.817ms     103.162us            37  
                                cudaGetDeviceProperties         0.02%     882.000us         0.02%     882.000us     882.000us       0.000us         0.00%       0.000us       0.000us             1  
                                            aten::relu_         0.01%     488.000us         0.02%     856.000us      50.353us       0.000us         0.00%      15.519ms     912.882us            17  
                                             aten::view         0.02%     677.000us         0.02%     677.000us      18.297us       0.000us         0.00%       0.000us       0.000us            37  
                                        cudaEventRecord         0.01%     400.000us         0.01%     400.000us       3.077us       0.000us         0.00%       0.000us       0.000us           130  
                           cudaStreamCreateWithPriority         0.01%     384.000us         0.01%     384.000us      96.000us       0.000us         0.00%       0.000us       0.000us             4  
                                       aten::threshold_         0.01%     224.000us         0.01%     368.000us      21.647us      15.519ms         1.49%      15.519ms     912.882us            17  
                                         cudaMemGetInfo         0.01%     277.000us         0.01%     277.000us      69.250us       0.000us         0.00%       0.000us       0.000us             4  
                                          aten::resize_         0.01%     266.000us         0.01%     266.000us       3.800us       0.000us         0.00%       0.000us       0.000us            70  
                                    aten::pixel_shuffle         0.00%      33.000us         0.01%     249.000us     249.000us       0.000us         0.00%       3.817ms       3.817ms             1  
                                   cudaEventElapsedTime         0.01%     219.000us         0.01%     219.000us      10.429us       0.000us         0.00%       0.000us       0.000us            21  
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFla...         0.00%     188.000us         0.00%     188.000us       0.013us       0.000us         0.00%       0.000us       0.000us         14544  
                                            aten::clone         0.00%      40.000us         0.00%     109.000us     109.000us       0.000us         0.00%       3.817ms       3.817ms             1  
                                            aten::zeros         0.00%      42.000us         0.00%      72.000us      72.000us       0.000us         0.00%       0.000us       0.000us             1  
                                            aten::copy_         0.00%      29.000us         0.00%      45.000us      45.000us       3.817ms         0.37%       3.817ms       3.817ms             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 4.733s
Self CUDA time total: 1.040s

Total number of params : 1332931
```

<br />

## EDSR depthwise separable convolution residual blocks
```bash
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                        model_inference         0.13%       5.775ms       100.00%        4.569s        4.569s       0.000us         0.00%     856.096ms     856.096ms             1  
                                           aten::conv2d         0.03%       1.371ms        98.60%        4.505s      67.246ms       0.000us         0.00%     792.398ms      11.827ms            67  
                                      aten::convolution         0.02%     779.000us        98.57%        4.504s      67.225ms       0.000us         0.00%     792.398ms      11.827ms            67  
                                     aten::_convolution         0.04%       1.787ms        98.55%        4.503s      67.214ms       0.000us         0.00%     792.398ms      11.827ms            67  
                                aten::cudnn_convolution         3.81%     174.275ms        97.10%        4.437s     126.772ms     695.072ms        81.19%     695.072ms      19.859ms            35  
                                        cudaMemsetAsync        40.45%        1.848s        40.45%        1.848s     462.114ms       0.000us         0.00%       0.000us       0.000us             4  
                                               cudaFree        37.34%        1.706s        37.34%        1.706s     341.228ms       0.000us         0.00%       0.000us       0.000us             5  
                                   cudaEventSynchronize         6.59%     301.003ms         6.59%     301.003ms      15.842ms       0.000us         0.00%       0.000us       0.000us            19  
                                       cudaLaunchKernel         5.60%     255.895ms         5.60%     255.895ms       6.847us       0.000us         0.00%       0.000us       0.000us         37371  
                                             cudaMalloc         5.55%     253.632ms         5.55%     253.632ms       3.294ms       0.000us         0.00%       0.000us       0.000us            77  
                            aten::thnn_conv_depthwise2d         0.01%     427.000us         1.35%      61.545ms       1.923ms       0.000us         0.00%      68.806ms       2.150ms            32  
                    aten::thnn_conv_depthwise2d_forward         0.04%       1.870ms         1.34%      61.118ms       1.910ms      68.806ms         8.04%      68.806ms       2.150ms            32  
                                            aten::empty         0.02%     796.000us         1.20%      54.841ms     783.443us       0.000us         0.00%       0.000us       0.000us            70  
                                              aten::mul         0.03%       1.330ms         1.17%      53.615ms       1.675ms      24.539ms         2.87%      24.539ms     766.844us            32  
                                          aten::reshape         0.01%     378.000us         0.09%       3.994ms     107.946us       0.000us         0.00%       3.816ms     103.135us            37  
                                          cudaHostAlloc         0.07%       3.416ms         0.07%       3.416ms       3.416ms       0.000us         0.00%       0.000us       0.000us             1  
                                    aten::pixel_shuffle         0.00%      34.000us         0.06%       2.963ms       2.963ms       0.000us         0.00%       3.816ms       3.816ms             1  
                                            aten::clone         0.00%      48.000us         0.06%       2.815ms       2.815ms       0.000us         0.00%       3.816ms       3.816ms             1  
                                       aten::empty_like         0.00%      14.000us         0.06%       2.702ms       2.702ms       0.000us         0.00%       0.000us       0.000us             1  
                                             aten::add_         0.04%       1.929ms         0.06%       2.529ms      48.635us      48.343ms         5.65%      48.343ms     929.673us            52  
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFla...         0.05%       2.290ms         0.05%       2.290ms       0.184us       0.000us         0.00%       0.000us       0.000us         12412  
                                cudaGetDeviceProperties         0.04%       1.953ms         0.04%       1.953ms       1.953ms       0.000us         0.00%       0.000us       0.000us             1  
                                    cudaStreamWaitEvent         0.02%       1.003ms         0.02%       1.003ms       0.080us       0.000us         0.00%       0.000us       0.000us         12468  
                                            aten::relu_         0.01%     501.000us         0.02%     988.000us      58.118us       0.000us         0.00%      15.520ms     912.941us            17  
                                         cudaMemGetInfo         0.02%     856.000us         0.02%     856.000us     214.000us       0.000us         0.00%       0.000us       0.000us             4  
                           cudaStreamCreateWithPriority         0.02%     808.000us         0.02%     808.000us     202.000us       0.000us         0.00%       0.000us       0.000us             4  
                                             aten::view         0.02%     779.000us         0.02%     779.000us      21.054us       0.000us         0.00%       0.000us       0.000us            37  
                                       aten::threshold_         0.01%     289.000us         0.01%     487.000us      28.647us      15.520ms         1.81%      15.520ms     912.941us            17  
                                        cudaEventRecord         0.01%     437.000us         0.01%     437.000us       3.612us       0.000us         0.00%       0.000us       0.000us           121  
                                   cudaFuncSetAttribute         0.01%     403.000us         0.01%     403.000us       0.263us       0.000us         0.00%       0.000us       0.000us          1535  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 4.570s
Self CUDA time total: 856.096ms

Total number of params : 304835
```
# Results
## EDSR using Standard Conv & using Depthwise separable Conv comparison
### CPU time (standard > Depthwise separable)
- Standard : 4.733s
- Depthwise separable : 4.570s

<br />

### GPU time (standard > Depthwise separable)
- Standard : 1.040s
- Depthwise separable : 856.096ms

<br />

### Total number of params (standard > Depthwise separable)
- Standard : 1332931
- Depthwise separable : 304835

### PSNR
<table>
    <tr>
        <td><center>Bicubic</center></td>
        <td><center>EDSR Standard (26.89 dB)</center></td>
        <td><center>EDSR Depthwise Separable (26.66 dB)</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="examples/sample_bicubic_x2.jpg"></center>
    	</td>
    	<td>
    		<center><img src="examples/EDSR_x2_16_64_Standard.jpg"></center>
    	</td>
      <td>
    		<center><img src="examples/EDSR_x2_16_64_Depthwise_separable.jpg"></center>
    	</td>
    </tr>
</table>