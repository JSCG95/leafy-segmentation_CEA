import numpy as np
import cv2
import pandas as pd
import glob
from scipy import ndimage as nd
from skimage.filters import roberts, sobel, scharr, prewitt, meijering, sato, hessian
from matplotlib import pyplot as plt

def extract_features(img_path):
  df = pd.DataFrame()
  # print(file)
  img = cv2.imread(img_path)

  B_0 = img[:,:,0].reshape(-1)
  df["B channel(RGB)"] = B_0

  G_1 = img[:,:,1].reshape(-1)
  df["G channel(RGB)"] = G_1

  R_2 = img[:,:,2].reshape(-1)
  df["R channel(RGB)"] = R_2


  HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  H = HSV_img[:,:,0].reshape(-1)
  df["H channel(HSV)"] = H
  S = HSV_img[:,:,1].reshape(-1)
  df["S channel(HSV)"] = S
  V = HSV_img[:,:,1].reshape(-1)
  df["V channel(HSV)"] = V


  LAB_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
  L = LAB_img[:,:,1].reshape(-1)
  df["L channel(LAB)"] = L
  A = LAB_img[:,:,1].reshape(-1)
  df["A channel(LAB)"] = A
  Bb = LAB_img[:,:,2].reshape(-1)
  df["B channel(LAB)"] = Bb



  #flat_img = img.reshape(-1)
  #df["Gray"] = flat_img

  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Use gray scale image to add texture features
  #CANNY EDGE
  edges = cv2.Canny(img, 100,200)   #Image, min and max values
  edges1 = edges.reshape(-1)
  df["Canny"] = edges1

  #ROBERTS EDGE
  edge_roberts = roberts(img)
  edge_roberts1 = edge_roberts.reshape(-1)
  df['Roberts'] = edge_roberts1

  #SOBEL
  edge_sobel = sobel(img)
  edge_sobel1 = edge_sobel.reshape(-1)
  df['Sobel'] = edge_sobel1

  #SCHARR
  edge_scharr = scharr(img)
  edge_scharr1 = edge_scharr.reshape(-1)
  df['Scharr'] = edge_scharr1

  #PREWITT
  edge_prewitt = prewitt(img)
  edge_prewitt1 = edge_prewitt.reshape(-1)
  df['Prewitt'] = edge_prewitt1

  flat_img = img.reshape(-1)
  df["Grayscale"] = flat_img

  return df