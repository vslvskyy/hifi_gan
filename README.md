# HiFi-GAN Architecture Implementation

This is the homework for the course [Deep Learning for Audio](https://github.com/markovka17/dla) at the [CS Faculty](https://cs.hse.ru/en/)
  of [HSE](https://www.hse.ru/en/).
  
 The aim of this work was to implement [HiFi-GAN](https://arxiv.org/pdf/2010.05646.pdf) architecture and train it on [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) datset.

## Installation guide

### Clone repository
```shell
git clone https://github.com/vslvskyy/hifi_gan
cd hifi_gan
```

### Install dependencies
```shell
pip install -r requirements.txt
```

###  Download model weights
```shell
wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm= \
$(wget --quiet --save-cookies /tmp/cookies.txt \
--keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=1-yVAp4SmcXWtb0aXYDhU57ppYN9Ogx0W' \
-O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-yVAp4SmcXWtb0aXYDhU57ppYN9Ogx0W" \
-O checkpoint.pth && \
rm -rf /tmp/cookies.txt
```

## Model Testing
You can find generated samples in `examples` directory. To reproduce this result run:

```shell
python test.py --data_path ./test_data --checkpoint_path ./checkpoint_path
```
