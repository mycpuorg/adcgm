# adcgm

```
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 sample_monitoring_hooks.py
W0119 10:27:00.158000 3412475 torch/distributed/run.py:851]
W0119 10:27:00.158000 3412475 torch/distributed/run.py:851] *****************************************
W0119 10:27:00.158000 3412475 torch/distributed/run.py:851] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
W0119 10:27:00.158000 3412475 torch/distributed/run.py:851] *****************************************
[Iter 100] Mem: 0.17GB / 0.19GB reserved, Peak: 0.17GB, Loss: -0.0882
[Iter 100] Mem: 0.17GB / 0.19GB reserved, Peak: 0.17GB, Loss: -0.1575
[Iter 100] Mem: 0.17GB / 0.19GB reserved, Peak: 0.18GB, Loss: -0.0969[Iter 100] Mem: 0.17GB / 0.19GB reserved, Peak: 0.18GB, Loss: -0.0985

[Iter 100] Mem: 0.17GB / 0.19GB reserved, Peak: 0.17GB, Loss: -0.0905
[Iter 100] Mem: 0.17GB / 0.19GB reserved, Peak: 0.17GB, Loss: -0.0672[Iter 100] Mem: 0.17GB / 0.19GB reserved, Peak: 0.18GB, Loss: -0.0956

[Iter 100] Mem: 0.17GB / 0.19GB reserved, Peak: 0.17GB, Loss: -0.0748
[Iter 200] Mem: 0.17GB / 0.19GB reserved, Peak: 0.17GB, Loss: -0.3124
[Iter 200] Mem: 0.17GB / 0.19GB reserved, Peak: 0.17GB, Loss: -0.1232
[Iter 200] Mem: 0.17GB / 0.19GB reserved, Peak: 0.18GB, Loss: -0.1427
[Iter 200] Mem: 0.17GB / 0.19GB reserved, Peak: 0.18GB, Loss: -0.1685
[Iter 200] Mem: 0.17GB / 0.19GB reserved, Peak: 0.18GB, Loss: -0.2198
[Iter 200] Mem: 0.17GB / 0.19GB reserved, Peak: 0.17GB, Loss: -0.2551
[Iter 200] Mem: 0.17GB / 0.19GB reserved, Peak: 0.17GB, Loss: -0.1975
[Iter 200] Mem: 0.17GB / 0.19GB reserved, Peak: 0.17GB, Loss: -0.0986
[Iter 300] Mem: 0.17GB / 0.19GB reserved, Peak: 0.17GB, Loss: -0.2662[Iter 300] Mem: 0.17GB / 0.19GB reserved, Peak: 0.17GB, Loss: -0.3495
[Iter 300] Mem: 0.17GB / 0.19GB reserved, Peak: 0.17GB, Loss: -0.1209

[Iter 300] Mem: 0.17GB / 0.19GB reserved, Peak: 0.18GB, Loss: -0.2990[Iter 300] Mem: 0.17GB / 0.19GB reserved, Peak: 0.18GB, Loss: -0.2348

[Iter 300] Mem: 0.17GB / 0.19GB reserved, Peak: 0.17GB, Loss: -0.1478[Iter 300] Mem: 0.17GB / 0.19GB reserved, Peak: 0.18GB, Loss: -0.2529

[Iter 300] Mem: 0.17GB / 0.19GB reserved, Peak: 0.17GB, Loss: -0.5508
[Iter 400] Mem: 0.17GB / 0.19GB reserved, Peak: 0.17GB, Loss: -0.2786[Iter 400] Mem: 0.17GB / 0.19GB reserved, Peak: 0.17GB, Loss: -0.4184[Iter 400] Mem: 0.17GB / 0.19GB reserved, Peak: 0.17GB, Loss: -0.4148


[Iter 400] Mem: 0.17GB / 0.19GB reserved, Peak: 0.18GB, Loss: -0.4279
[Iter 400] Mem: 0.17GB / 0.19GB reserved, Peak: 0.18GB, Loss: -0.3445[Iter 400] Mem: 0.17GB / 0.19GB reserved, Peak: 0.18GB, Loss: -0.5623
....
```
