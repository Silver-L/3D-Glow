# 3D Glow

## Requirements
```
tensorflow-gpu==1.8.0
keras==2.2.0
pillow==5.2.0
toposort==1.5
horovod==0.13.8
```

## Training 3D Glow

### Generate Dataset
```
python generate_tfrc.py --data_list [] --outdir [] --imagesize []
```

### training
```
python train.py --data_dir []
```

## Reference
1, https://github.com/openai/glow \
2, https://arxiv.org/pdf/1807.03039.pdf