import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
import csv
import numpy as np
import argparse
from skimage.util import random_noise


def idealFilterLP(image, D0):
    FFTCenterImage = np.fft.fftshift(np.fft.fft2(image))
    imgShape = image.size
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            distance = sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)
            if distance < D0:
                base[y,x] = 1

    LowPassCenter = FFTCenterImage * base
    LowPass = np.fft.ifftshift(LowPassCenter)
    image = np.abs(np.fft.ifft2(LowPass))
    image1 = Image.fromarray(np.uint8(image))
    return image1

def idealFilterHP(image, D0):
    FFTCenterImage = np.fft.fftshift(np.fft.fft2(image))
    imgShape = image.size
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            distance = sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)
            if distance < D0:
                base[y,x] = 0

    HighPassCenter = FFTCenterImage * base
    HighPass= np.fft.ifftshift(HighPassCenter)
    image = np.abs(np.fft.ifft2(HighPass))
    image1 = Image.fromarray(np.uint8(image))
    return image1
    
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    device = torch.device('cuda')
    parser.add_argument('-imageFolder', default='CroppedImages',type=str)
    parser.add_argument('-modelPath',  default='Model/D-NetPAD_Model.pth',type=str)
    parser.add_argument('-perturbationType', default='LowPass',type=str, help='LowPass, HighPass, GaussianNoise, SaltPepper')
    # parser.add_argument('-perturbationAmt', default='',type=int, help='')
    args = parser.parse_args()


    # Load weights of single binary DesNet121 model
    weights = torch.load(args.modelPath)
    DNetPAD = models.densenet161(pretrained=True)
    num_ftrs = DNetPAD.classifier.in_features
    DNetPAD.classifier = nn.Linear(num_ftrs, 2)
    DNetPAD.load_state_dict(weights['state_dict'])
    DNetPAD = DNetPAD.to(device)
    DNetPAD.eval()


    # Transformation specified for the pre-processing
    transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])

    imagesScores=[]

    imageFiles = glob.glob(os.path.join(args.imageFolder,'*.jpg'))
    for imgFile in imageFiles:

            # Read the segmented image
            image = Image.open(imgFile)
            
            # Apply perturbations in the input image
            if args.perturbationType == 'LowPass':
                image = idealFilterLP(image,20)
            elif args.perturbationType == 'HighPass':
                image = idealFilterHP(image,5)
            elif args.perturbationType == 'GaussianNoise':
                image = np.asarray(image)
                noise_img = random_noise(image, mode='gaussian',var=0.001)
                image = Image.fromarray(np.uint8(noise_img*255))
            elif args.perturbationType == 'SaltPepper':
                image = np.asarray(image)
                noise_img = random_noise(image, mode='s&p',amount=0.005)
                image = Image.fromarray(np.uint8(noise_img*255))

            # Image transformation
            tranformImage = transform(image)
            image.close()
            tranformImage = tranformImage.repeat(3, 1, 1) # for NIR images having one channel
            tranformImage = tranformImage[0:3,:,:].unsqueeze(0)
            tranformImage = tranformImage.to(device)

            # Output from single binary CNN model
            output = DNetPAD(tranformImage)
            PAScore = output.detach().cpu().numpy()[:, 1]

            # Normalization of output score between [0,1]
            PAScore = np.minimum(np.maximum((PAScore+15)/35,0),1)
            imagesScores.append([imgFile, PAScore[0]])


    # Writing the scores in the csv file
    with open(os.path.join(args.imageFolder,'Scores.csv'),'w',newline='') as fout:
        writer = csv.writer(fout)
        writer.writerows(imagesScores)
