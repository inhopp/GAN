# GAN
GAN from scartch (pytorch)

| Epoch 0 | Epoch 50 | Epoch 100 | Epoch 150 | Epoch 200 |
|:-:|:-:|:-:|:-:|:-:|
| ![data0](https://user-images.githubusercontent.com/96368476/215316520-03512d96-1d3b-4eae-b16a-30c7e042c5fc.png) | ![data49](https://user-images.githubusercontent.com/96368476/215316070-7c6587ba-85fb-40ee-8c26-f777ce02f87e.png) | ![data99](https://user-images.githubusercontent.com/96368476/215316071-b530d879-19bb-4fe0-bc9d-e963eb57bc96.png) | ![data149](https://user-images.githubusercontent.com/96368476/215316072-6c7be07b-9c12-4541-8bf2-a1ff6011ba18.png) | ![data199](https://user-images.githubusercontent.com/96368476/215316073-37175110-0f99-4793-a49e-23422cad5a86.png) |


## Repository Directory 

``` python 
├── GAN
        ├── datasets
        │     └── mnist
        ├── data.py
        ├── option.py
        ├── model.py
        ├── train.py
        └── README.md
```

- `data.py` : data load (download mnist)
- `data/dataset.py` : data preprocess & get item
- `model.py` : Define block and construct Model
- `option.py` : Environment setting

<br>


## Tutoral

### Clone repo and install depenency

``` python
# Clone this repo and install dependency
git clone https://github.com/inhopp/GAN.git
```

<br>


### train
``` python
python3 train.py
    --device {}(default: cpu) \
    --input_size{}(default: 28) \
    --lr {}(default: 0.0002) \
    --n_epoch {}(default: 200) \
    --num_workers {}(default: 4) \
    --batch_size {}(default: 64) \
```


<br>


#### Main Reference
https://github.com/happy-jihye/Awesome-GAN-Papers/blob/main/gan/gan.ipynb