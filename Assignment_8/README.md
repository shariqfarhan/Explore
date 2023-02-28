# Receptive Field Calculations

<img width="1060" alt="image" src="https://user-images.githubusercontent.com/57046534/221738707-234fd981-a2c2-4c55-8bdd-60d2e34778b4.png">


```
Input_features	padding	kernel	stride	dilation	jump_In	jump_out	RF_In	RF_Out	Output_Feature	Input_Channel	Output_Channels	kernel_adjusted	Convolution Type
32	1	3	1	1	1	1	1	3	32	3	64	3	Normal
32	1	3	1	1	1	1	3	5	32	64	128	3	Normal
32	0	2	2	1	1	2	5	6	16	128	128	2	MaxPool
16	1	3	1	1	2	2	6	10	16	128	128	3	Normal
16	1	3	1	1	2	2	10	14	16	128	128	3	Normal
16	1	3	1	1	2	2	14	18	16	128	256	3	Normal
16	0	2	2	1	2	4	18	20	8	256	256	2	MaxPool
8	1	3	1	1	4	4	20	28	8	256	512	3	Normal
8	0	2	2	1	4	8	28	32	4	512	512	2	MaxPool
4	1	3	1	1	8	8	32	48	4	512	512	3	Normal
4	1	3	1	1	8	8	48	64	4	512	512	3	Normal
4	0	4	1	1	8	8	64	88	1	512	512	4	MaxPool
```

# Training Logs
```
EPOCH: 1
Batch_id=781 Loss=1.61540 Accuracy=43.16: 100%|██████████| 782/782 [00:15<00:00, 49.21it/s]

Test set: Average loss: 1.4431, Accuracy: 5235/10000 (52.35%)
EPOCH: 2
Batch_id=781 Loss=1.32396 Accuracy=56.02: 100%|██████████| 782/782 [00:15<00:00, 49.37it/s]

Test set: Average loss: 1.0299, Accuracy: 6779/10000 (67.79%)
EPOCH: 3
Batch_id=781 Loss=1.21953 Accuracy=61.72: 100%|██████████| 782/782 [00:15<00:00, 49.34it/s]

Test set: Average loss: 1.1159, Accuracy: 6962/10000 (69.62%)
EPOCH: 4
Batch_id=781 Loss=1.02420 Accuracy=67.60: 100%|██████████| 782/782 [00:15<00:00, 49.00it/s]

Test set: Average loss: 1.3809, Accuracy: 6536/10000 (65.36%)
EPOCH: 5
Batch_id=781 Loss=0.89883 Accuracy=71.04: 100%|██████████| 782/782 [00:15<00:00, 49.53it/s]

Test set: Average loss: 1.0891, Accuracy: 7071/10000 (70.71%)
EPOCH: 6
Batch_id=781 Loss=0.76692 Accuracy=74.72: 100%|██████████| 782/782 [00:15<00:00, 49.21it/s]

Test set: Average loss: 0.5437, Accuracy: 8170/10000 (81.70%)
EPOCH: 7
Batch_id=781 Loss=0.66969 Accuracy=77.59: 100%|██████████| 782/782 [00:15<00:00, 49.50it/s]

Test set: Average loss: 0.6329, Accuracy: 8080/10000 (80.80%)
EPOCH: 8
Batch_id=781 Loss=0.60968 Accuracy=79.26: 100%|██████████| 782/782 [00:15<00:00, 49.59it/s]

Test set: Average loss: 0.4856, Accuracy: 8443/10000 (84.43%)
EPOCH: 9
Batch_id=781 Loss=0.54375 Accuracy=81.42: 100%|██████████| 782/782 [00:15<00:00, 49.50it/s]

Test set: Average loss: 0.4593, Accuracy: 8541/10000 (85.41%)
EPOCH: 10
Batch_id=781 Loss=0.50340 Accuracy=82.84: 100%|██████████| 782/782 [00:15<00:00, 49.40it/s]

Test set: Average loss: 0.4216, Accuracy: 8640/10000 (86.40%)
EPOCH: 11
Batch_id=781 Loss=0.47899 Accuracy=83.41: 100%|██████████| 782/782 [00:15<00:00, 49.28it/s]

Test set: Average loss: 0.4321, Accuracy: 8577/10000 (85.77%)
EPOCH: 12
Batch_id=781 Loss=0.44418 Accuracy=84.69: 100%|██████████| 782/782 [00:15<00:00, 49.47it/s]

Test set: Average loss: 0.3546, Accuracy: 8879/10000 (88.79%)
EPOCH: 13
Batch_id=781 Loss=0.42001 Accuracy=85.34: 100%|██████████| 782/782 [00:15<00:00, 49.93it/s]

Test set: Average loss: 0.3528, Accuracy: 8855/10000 (88.55%)
EPOCH: 14
Batch_id=781 Loss=0.38036 Accuracy=86.78: 100%|██████████| 782/782 [00:15<00:00, 49.64it/s]

Test set: Average loss: 0.3328, Accuracy: 8944/10000 (89.44%)
EPOCH: 15
Batch_id=781 Loss=0.34496 Accuracy=87.88: 100%|██████████| 782/782 [00:15<00:00, 49.32it/s]

Test set: Average loss: 0.3185, Accuracy: 8990/10000 (89.90%)
EPOCH: 16
Batch_id=781 Loss=0.31726 Accuracy=88.96: 100%|██████████| 782/782 [00:15<00:00, 49.34it/s]

Test set: Average loss: 0.2839, Accuracy: 9098/10000 (90.98%)
EPOCH: 17
Batch_id=781 Loss=0.29380 Accuracy=89.79: 100%|██████████| 782/782 [00:15<00:00, 49.97it/s]

Test set: Average loss: 0.2784, Accuracy: 9109/10000 (91.09%)
EPOCH: 18
Batch_id=781 Loss=0.27083 Accuracy=90.64: 100%|██████████| 782/782 [00:15<00:00, 49.56it/s]

Test set: Average loss: 0.2620, Accuracy: 9146/10000 (91.46%)
EPOCH: 19
Batch_id=781 Loss=0.25079 Accuracy=91.27: 100%|██████████| 782/782 [00:15<00:00, 49.59it/s]

Test set: Average loss: 0.2607, Accuracy: 9171/10000 (91.71%)
EPOCH: 20
Batch_id=781 Loss=0.23885 Accuracy=91.75: 100%|██████████| 782/782 [00:15<00:00, 50.05it/s]

Test set: Average loss: 0.2570, Accuracy: 9191/10000 (91.91%)
```
