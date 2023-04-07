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

Without OneCycle Policy

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

With OneCycle Policy

```
EPOCH: 1 (LR: 0.0004)
Loss=2.3673577308654785 Batch_id=97 Accuracy=25.96: 100%|██████████| 98/98 [00:27<00:00,  3.50it/s]
Test set: Average loss: 1.5617, Accuracy: 4201/10000 (42.01%)

EPOCH: 2 (LR: 0.002323926380368098)
Loss=2.1892008781433105 Batch_id=97 Accuracy=39.57: 100%|██████████| 98/98 [00:28<00:00,  3.44it/s]
Test set: Average loss: 1.4020, Accuracy: 4825/10000 (48.25%)

EPOCH: 3 (LR: 0.004247852760736196)
Loss=2.2367911338806152 Batch_id=97 Accuracy=41.77: 100%|██████████| 98/98 [00:28<00:00,  3.42it/s]
Test set: Average loss: 2.0183, Accuracy: 4563/10000 (45.63%)

EPOCH: 4 (LR: 0.0061717791411042945)
Loss=2.0869522094726562 Batch_id=97 Accuracy=45.66: 100%|██████████| 98/98 [00:27<00:00,  3.51it/s]
Test set: Average loss: 1.4928, Accuracy: 5012/10000 (50.12%)

EPOCH: 5 (LR: 0.008095705521472393)
Loss=2.205122470855713 Batch_id=97 Accuracy=48.48: 100%|██████████| 98/98 [00:28<00:00,  3.45it/s]
Test set: Average loss: 1.1331, Accuracy: 6135/10000 (61.35%)

EPOCH: 6 (LR: 0.009994629452201934)
Loss=2.060209274291992 Batch_id=97 Accuracy=51.03: 100%|██████████| 98/98 [00:28<00:00,  3.44it/s]
Test set: Average loss: 1.0601, Accuracy: 6439/10000 (64.39%)

EPOCH: 7 (LR: 0.009468315767991408)
Loss=1.9966624975204468 Batch_id=97 Accuracy=56.19: 100%|██████████| 98/98 [00:28<00:00,  3.49it/s]
Test set: Average loss: 1.0376, Accuracy: 6493/10000 (64.93%)

EPOCH: 8 (LR: 0.00894200208378088)
Loss=1.7902780771255493 Batch_id=97 Accuracy=58.12: 100%|██████████| 98/98 [00:28<00:00,  3.45it/s]
Test set: Average loss: 1.0328, Accuracy: 6700/10000 (67.00%)

EPOCH: 9 (LR: 0.008415688399570355)
Loss=1.8230100870132446 Batch_id=97 Accuracy=59.86: 100%|██████████| 98/98 [00:28<00:00,  3.46it/s]
Test set: Average loss: 0.7618, Accuracy: 7344/10000 (73.44%)

EPOCH: 10 (LR: 0.007889374715359828)
Loss=1.7145123481750488 Batch_id=97 Accuracy=61.09: 100%|██████████| 98/98 [00:28<00:00,  3.48it/s]
Test set: Average loss: 0.7796, Accuracy: 7361/10000 (73.61%)

EPOCH: 11 (LR: 0.007363061031149302)
Loss=1.5972459316253662 Batch_id=97 Accuracy=63.01: 100%|██████████| 98/98 [00:28<00:00,  3.46it/s]
Test set: Average loss: 0.7783, Accuracy: 7386/10000 (73.86%)

EPOCH: 12 (LR: 0.006836747346938775)
Loss=1.5847711563110352 Batch_id=97 Accuracy=64.00: 100%|██████████| 98/98 [00:28<00:00,  3.46it/s]
Test set: Average loss: 0.7419, Accuracy: 7510/10000 (75.10%)

EPOCH: 13 (LR: 0.00631043366272825)
Loss=1.5906178951263428 Batch_id=97 Accuracy=65.37: 100%|██████████| 98/98 [00:28<00:00,  3.47it/s]
Test set: Average loss: 0.7626, Accuracy: 7453/10000 (74.53%)

EPOCH: 14 (LR: 0.0057841199785177225)
Loss=1.6446447372436523 Batch_id=97 Accuracy=65.99: 100%|██████████| 98/98 [00:28<00:00,  3.44it/s]
Test set: Average loss: 0.7080, Accuracy: 7646/10000 (76.46%)

EPOCH: 15 (LR: 0.005257806294307196)
Loss=1.6105029582977295 Batch_id=97 Accuracy=67.37: 100%|██████████| 98/98 [00:28<00:00,  3.46it/s]
Test set: Average loss: 0.6350, Accuracy: 7857/10000 (78.57%)

EPOCH: 16 (LR: 0.004731492610096671)
Loss=1.6064486503601074 Batch_id=97 Accuracy=67.80: 100%|██████████| 98/98 [00:28<00:00,  3.46it/s]
Test set: Average loss: 0.6398, Accuracy: 7829/10000 (78.29%)

EPOCH: 17 (LR: 0.004205178925886144)
Loss=1.3985605239868164 Batch_id=97 Accuracy=68.93: 100%|██████████| 98/98 [00:28<00:00,  3.42it/s]
Test set: Average loss: 0.6133, Accuracy: 7920/10000 (79.20%)

EPOCH: 18 (LR: 0.003678865241675618)
Loss=1.4219648838043213 Batch_id=97 Accuracy=69.65: 100%|██████████| 98/98 [00:28<00:00,  3.47it/s]
Test set: Average loss: 0.5675, Accuracy: 8075/10000 (80.75%)

EPOCH: 19 (LR: 0.0031525515574650905)
Loss=1.409639596939087 Batch_id=97 Accuracy=70.64: 100%|██████████| 98/98 [00:28<00:00,  3.48it/s]
Test set: Average loss: 0.5829, Accuracy: 8053/10000 (80.53%)

EPOCH: 20 (LR: 0.002626237873254565)
Loss=1.3928968906402588 Batch_id=97 Accuracy=71.71: 100%|██████████| 98/98 [00:28<00:00,  3.47it/s]
Test set: Average loss: 0.5361, Accuracy: 8213/10000 (82.13%)

EPOCH: 21 (LR: 0.0020999241890440386)
Loss=1.464630126953125 Batch_id=97 Accuracy=71.56: 100%|██████████| 98/98 [00:28<00:00,  3.45it/s]
Test set: Average loss: 0.5421, Accuracy: 8204/10000 (82.04%)

EPOCH: 22 (LR: 0.0015736105048335131)
Loss=1.3937873840332031 Batch_id=97 Accuracy=72.57: 100%|██████████| 98/98 [00:28<00:00,  3.46it/s]
Test set: Average loss: 0.5167, Accuracy: 8303/10000 (83.03%)

EPOCH: 23 (LR: 0.0010472968206229859)
Loss=1.3393802642822266 Batch_id=97 Accuracy=73.34: 100%|██████████| 98/98 [00:28<00:00,  3.47it/s]
Test set: Average loss: 0.5087, Accuracy: 8314/10000 (83.14%)

EPOCH: 24 (LR: 0.0005209831364124604)
Loss=1.3247623443603516 Batch_id=97 Accuracy=73.69: 100%|██████████| 98/98 [00:28<00:00,  3.47it/s]
Test set: Average loss: 0.4964, Accuracy: 8347/10000 (83.47%)
```
