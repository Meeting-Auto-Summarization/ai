import gdown
import os

if __name__ == '__main__':
    url = 'https://drive.google.com/u/0/uc?id=1El-HnebxD_9cU8tmEsPMEqSzwX6oh26h&export=download'
    output = '.models/pytorch_model.bin'
    
    os.mkdir('./.models')
    gdown.download(url, output)