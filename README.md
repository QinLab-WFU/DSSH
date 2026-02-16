Source code for TKDE 2026 paper “Deep Stochastic Spherical Hashing with von Mises-Fisher Distributions for Cross-Modal
Retrieval”.
## Training	
### Processing dataset

Refer to [DSPH](https://github.com/QinLab-WFU/DSPH)

### Download CLIP pretrained model
Pretrained model will be found in the 30 lines of [CLIP/clip/clip.py](CLIP/clip/clip.py). This code is based on the "ViT-B/32".

You should copy ViT-B-32.pt to this dir.

### Start
```python
python main.py

