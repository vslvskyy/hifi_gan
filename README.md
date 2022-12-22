# HiFi-GAN Architecture Implementation

This is the homework for the course [Deep Learning for Audio](https://github.com/markovka17/dla) at the [CS Faculty](https://cs.hse.ru/en/)
  of [HSE](https://www.hse.ru/en/).

 The aim of this work was to implement [HiFi-GAN](https://arxiv.org/pdf/2010.05646.pdf) architecture and train it on [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) dataset.

## Installation guide

### Clone repository
```shell
git clone https://github.com/vslvskyy/hifi_gan
cd hifi_gan
mkdir results
```

### Install dependencies
```shell
pip install -r requirements.txt
```

## Model Testing

You can find generated samples in the `examples` directory. To reproduce the results do the following:

###  Download model weights
```shell
wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm= \
$(wget --quiet --save-cookies /tmp/cookies.txt \
--keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=1-y6dL8jwJH-pM0ZidBRkuQgxfinN1Rh_' \
-O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-y6dL8jwJH-pM0ZidBRkuQgxfinN1Rh_" \
-O checkpoint_epoch100.pth && \
rm -rf /tmp/cookies.txt
```

### Generate wavs

```shell
python test.py \
    --data_path ./test_dataset \
    --checkpoint_path ./checkpoint_epoch100.pth \
    --results_dir_path ./results
```

Now samples are avaliable at `./results`. You may display them like this:

```python
from IPython import display

display.Audio("./results/result_wav_0.wav")
display.Audio("./results/result_wav_1.wav")
display.Audio("./results/result_wav_2.wav")
```

## Model Training

### Download LJSpeech dataset

```shell
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null
mkdir data
tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
mv LJSpeech-1.1 data/LJSpeech-1.1
```

### Train model

```shell
python train.py \
    --train_data_path ./data/LJSpeech-1.1 \
    --test_data_path ./test_dataset \
    --checkpoint_path ./ \
    --wandb_log  # if wandb logging is needed
```
