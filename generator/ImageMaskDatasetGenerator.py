# Copyright 2024 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 2024/07/17
# ImageMaskDatasetGenerator:


import os
import io
import sys
import glob
import numpy as np
import cv2

import math
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

import shutil
import traceback

class ImageMaskDatasetGenerator:

  def __init__(self, input_images_dir, 
                input_masks_dir,
                output_dir, 
                square_resize=True,
                augmentation=False):

     self.input_images_dir = input_images_dir
     self.input_masks_dir  = input_masks_dir
     self.output_dir = output_dir
     self.seed      = 137
     self.W         = 512
     self.H         = 512
     self.augmentation = augmentation
     self.square_resize   = square_resize
     if self.augmentation:
      self.hflip    = True
      self.vflip    = True
      self.rotation = True
      self.ANGLES   = [90, 180, 270, ]

      self.resize = False
      self.resize_ratios = [0.6, 0.8 ]

      self.deformation=False
      self.alpha    = 1300
      self.sigmoids = [8.0, 10.0]
          
      self.distortion=False
      self.gaussina_filer_rsigma = 40
      self.gaussina_filer_sigma  = 0.5
      self.distortions           = [0.02,0.03]
      self.rsigma = "sigma"  + str(self.gaussina_filer_rsigma)
      self.sigma  = "rsigma" + str(self.gaussina_filer_sigma)     

      self.barrel_distortion = True
      self.radius     = 0.3
      self.amount     = 0.3
      self.centers    = [(0.3, 0.3), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.7, 0.7)]

      self.pincussion_distortion = False
      self.pinc_radius  = 0.3
      self.pinc_amount  = -0.3
      self.pinc_centers = [(0.3, 0.3), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.7, 0.7)]

     if os.path.exists(self.output_dir):
        shutil.rmtree(self.output_dir)

     if not os.path.exists(self.output_dir):
        os.makedirs(self.output_dir)

     self.output_images_dir = self.output_dir +  "/images"
     self.output_masks_dir  = self.output_dir +  "/masks"
     os.makedirs(self.output_images_dir)
     os.makedirs(self.output_masks_dir)

  def generate(self):
    print("=== generate ")
    self.image_index = 10000
    self.mask_index  = 10000

    image_files = glob.glob(self.input_images_dir + "/*.jpg")
    image_files = sorted(image_files)
     
    mask_files   = glob.glob(self.input_masks_dir + "/*.png")
    mask_files   = sorted(mask_files)
    num_image_files= len(image_files)
    num_mask_files = len(mask_files)
    print("=== num_image_files {}".format(num_image_files))
    print("=== num_mask_files   {}".format(num_mask_files))
      
    if num_image_files != num_mask_files:
       error = "Unmatched image_files and mask_files"
       print(">>> Error {}".format(error))
       raise Exception(error)
     
    for i in range(num_image_files):
        image_file = image_files[i] 
        mask_file  = mask_files[i]
        basename   = os.path.basename(mask_file)
        mask_name  = basename.replace(".png", ".jpg") 
        print("---image_file {}".format(image_file))
        image_name = os.path.basename(image_file)
        image = cv2.imread(image_file)
        image_filepath = os.path.join(self.output_images_dir, image_name)
        if self.square_resize:
          #image = self.resize_to_512x512(image, ismask=False)
          image = cv2.resize(image, (self.W, self.H))
        cv2.imwrite(image_filepath , image)
        print("=== Saved {}".format(image_filepath))

        if self.augmentation:
            self.augment(image, image_name, self.output_images_dir, border=(0, 0, 0), mask=False)      
    
        mask = cv2.imread(mask_file)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
        mask_filepath = os.path.join(self.output_masks_dir,  mask_name)
        if self.square_resize:
          #  mask = self.resize_to_512x512(mask, ismask=True)
          mask = cv2.resize(mask, (self.W, self.H))

        cv2.imwrite(mask_filepath, mask)

        print("=== Saved {}".format(mask_filepath))
          
        if self.augmentation:
            self.augment(mask, mask_name, self.output_masks_dir, border=(0, 0, 0), mask=True)      
      

  def resize_to_512x512(self, image, ismask=False):
    h, w = image.shape[:2]
    pixel = image[2][2]

    RESIZE = h
    if w > h:
      RESIZE = w
    # 1. Create a black background
    if ismask:
      background = np.zeros((RESIZE, RESIZE, 1),  np.uint8)
    else:
      background = np.ones((RESIZE, RESIZE,  3),  np.uint8)*pixel

    x = int((RESIZE - w)/2)
    y = int((RESIZE - h)/2)
    # 2. Paste the image to the background 
    if image.ndim == 2:
      image = np.expand_dims(image, axis=-1)
    background[y:y+h, x:x+w] = image
    # 3. Resize the background to (512x512)
    resized = cv2.resize(background, (self.W, self.H))

    return resized

  def augment(self, image, basename, output_dir, border=(0, 0, 0), mask=False):
    border = image[2][2].tolist()
  
    print("---- border {}".format(border))
    if self.hflip:
      flipped = self.horizontal_flip(image)
      output_filepath = os.path.join(output_dir, "hflipped_" + basename)
      cv2.imwrite(output_filepath, flipped)
      print("--- Saved {}".format(output_filepath))

    if self.vflip:
      flipped = self.vertical_flip(image)
      output_filepath = os.path.join(output_dir, "vflipped_" + basename)
      cv2.imwrite(output_filepath, flipped)
      print("--- Saved {}".format(output_filepath))

    if self.rotation:
      self.rotate(image, basename, output_dir, border)

    if self.deformation:
      self.deform(image, basename, output_dir)

    if self.distortion:
      self.distort(image, basename, output_dir)

    if self.resize:
      self.shrink(image, basename, output_dir, mask)

    if self.barrel_distortion:
      self.barrel_distort(image, basename, output_dir)

    if self.pincussion_distortion:
      self.pincussion_distort(image, basename, output_dir)


  def horizontal_flip(self, image): 
    print("shape image {}".format(image.shape))
    if len(image.shape)==3:
      return  image[:, ::-1, :]
    else:
      return  image[:, ::-1, ]

  def vertical_flip(self, image):
    if len(image.shape) == 3:
      return image[::-1, :, :]
    else:
      return image[::-1, :, ]

  def rotate(self, image, basename, output_dir, border):
    for angle in self.ANGLES:      
      center = (self.W/2, self.H/2)
      rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)

      rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(self.W, self.H), borderValue=border)
      output_filepath = os.path.join(output_dir, "rotated_" + str(angle) + "_" + basename)
      cv2.imwrite(output_filepath, rotated_image)
      print("--- Saved {}".format(output_filepath))

  def deform(self, image, basename, output_dir): 
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    random_state = np.random.RandomState(self.seed)

    shape = image.shape
    for sigmoid in self.sigmoids:
      dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigmoid, mode="constant", cval=0) * self.alpha
      dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigmoid, mode="constant", cval=0) * self.alpha
      #dz = np.zeros_like(dx)

      x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
      indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

      deformed_image = map_coordinates(image, indices, order=1, mode='nearest')  
      deformed_image = deformed_image.reshape(image.shape)

      image_filename = "deformed" + "_" + str(self.alpha) + "_" +str(sigmoid) + "_" + basename
      image_filepath  = os.path.join(output_dir, image_filename)
      cv2.imwrite(image_filepath, deformed_image)

  def distort(self, image, basename, output_dir):
    shape = (image.shape[1], image.shape[0])
    (w, h) = shape
    xsize = w
    if h>w:
      xsize = h
    # Resize original img to a square image
    resized = cv2.resize(image, (xsize, xsize))
 
    shape   = (xsize, xsize)
 
    t = np.random.normal(size = shape)
    for size in self.distortions:
      filename = "distorted_" + str(size) + "_" + self.sigma + "_" + self.rsigma + "_" + basename
      output_file = os.path.join(output_dir, filename)    
      dx = gaussian_filter(t, self.gaussina_filer_rsigma, order =(0,1))
      dy = gaussian_filter(t, self.gaussina_filer_rsigma, order =(1,0))
      sizex = int(xsize*size)
      sizey = int(xsize*size)
      dx *= sizex/dx.max()
      dy *= sizey/dy.max()

      image = gaussian_filter(image, self.gaussina_filer_sigma)

      yy, xx = np.indices(shape)
      xmap = (xx-dx).astype(np.float32)
      ymap = (yy-dy).astype(np.float32)

      distorted = cv2.remap(resized, xmap, ymap, cv2.INTER_LINEAR)
      distorted = cv2.resize(distorted, (w, h))
      cv2.imwrite(output_file, distorted)
      print("=== Saved distorted image file{}".format(output_file))

  # This method has been taken from the following stackoverflow website. 
  # https://stackoverflow.com/questions/59776772/python-opencv-how-to-apply-radial-barrel-distortion

  def barrel_distort(self, image, basename, output_dir):
    distorted_image  = image
    #(h, w, _) = image.shape
    h = image.shape[0]
    w = image.shape[1]
    # set up the x and y maps as float32
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)

    scale_x = 1
    scale_y = 1
    index = 100
    for center in self.centers:
      index += 1
      (ox, oy) = center
      center_x = w * ox
      center_y = h * oy
      radius = w * self.radius
      amount = self.amount   
      # negative values produce pincushion
 
      # create map with the barrel pincushion distortion formula
      for y in range(h):
        delta_y = scale_y * (y - center_y)
        for x in range(w):
          # determine if pixel is within an ellipse
          delta_x = scale_x * (x - center_x)
          distance = delta_x * delta_x + delta_y * delta_y
          if distance >= (radius * radius):
            map_x[y, x] = x
            map_y[y, x] = y
          else:
            factor = 1.0
            if distance > 0.0:
                factor = math.pow(math.sin(math.pi * math.sqrt(distance) / radius / 2), amount)
            map_x[y, x] = factor * delta_x / scale_x + center_x
            map_y[y, x] = factor * delta_y / scale_y + center_y
            
       # do the remap
      distorted_image = cv2.remap(distorted_image, map_x, map_y, cv2.INTER_LINEAR)
      if distorted_image.ndim == 2:
        distorted_image  = np.expand_dims(distorted_image, axis=-1) 

      image_filename = "barrdistorted_" + str(index) + "_" + str(self.radius) + "_"  + str(self.amount) + "_" + basename

      image_filepath  = os.path.join(output_dir, image_filename)
      cv2.imwrite(image_filepath, distorted_image)
      print("=== Saved {}".format(image_filepath))


  def pincussion_distort(self, image, basename, output_dir):
    distorted_image  = image
    h = image.shape[0]
    w = image.shape[1]

    # set up the x and y maps as float32
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)

    scale_x = 1
    scale_y = 1
    index = 100
    for center in self.pinc_centers:
      index += 1
      (ox, oy) = center
      center_x = w * ox
      center_y = h * oy
      radius = w * self.pinc_radius
      amount = self.pinc_amount   
      # negative values produce pincushion
 
      # create map with the barrel pincushion distortion formula
      for y in range(h):
        delta_y = scale_y * (y - center_y)
        for x in range(w):
          # determine if pixel is within an ellipse
          delta_x = scale_x * (x - center_x)
          distance = delta_x * delta_x + delta_y * delta_y
          if distance >= (radius * radius):
            map_x[y, x] = x
            map_y[y, x] = y
          else:
            factor = 1.0
            if distance > 0.0:
                factor = math.pow(math.sin(math.pi * math.sqrt(distance) / radius / 2), amount)
            map_x[y, x] = factor * delta_x / scale_x + center_x
            map_y[y, x] = factor * delta_y / scale_y + center_y
            

       # do the remap
      distorted_image = cv2.remap(distorted_image, map_x, map_y, cv2.INTER_LINEAR)
      if distorted_image.ndim == 2:
        distorted_image  = np.expand_dims(distorted_image, axis=-1) 

      image_filename = "pincdistorted_" + str(index) + "_" + str(self.pinc_radius) + "_"  + str(self.pinc_amount) + "_" + basename

      image_filepath  = os.path.join(output_dir, image_filename)
      cv2.imwrite(image_filepath, distorted_image)
      print("=== Saved {}".format(image_filepath))


if __name__ == "__main__":
  try:
      # 1 Create image-mask dataset for training and validation
      # Resize all images and masks to 512x512
      # Save them as JPEG files
      
      input_images_dir = "./BloodImageSetS6NucSeg/"
      input_masks_dir  = "./BloodImageSetS6NucSeg_masks/" 

      output_dir     = "./White-Blood-Cell-master"

      augmentation   = False
      generator = ImageMaskDatasetGenerator(input_images_dir,
                                            input_masks_dir,
                                             
                                           output_dir,
                                           square_resize = True,
                                           augmentation=augmentation)
      generator.generate()
  except:
    traceback.print_exc()
