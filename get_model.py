import gdown
import os
import shutil

if __name__ == '__main__':
    url = 'https://drive.google.com/u/0/uc?id=16xugNY0Tel-2-sIO4WgZAr0Kp5OiU6D_&export=download'
    output = '.models/pytorch_model.bin'
    
    if os.path.exists('./.models'):
        shutil.rmtree('./.models')
    os.mkdir('./.models')
    gdown.download(url, output)