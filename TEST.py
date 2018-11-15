from PIL import Image
import csv
import glob
import numpy as np
import os
import re
import scipy.io
import time
import tensorflow as tf

from MODEL import model
from PSNR import psnr

DATA_PATH = "./data/test/"

# filelist
folder_list = glob.glob(os.path.join(DATA_PATH, 'Set*'))
test_list = []
for folder_path in folder_list:
  print('>> folder : ', folder_path)
  l = glob.glob(os.path.join(folder_path, "*"))
  print("     total samples : ", len(l))
  l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
  print("     total images : ", len(l))
  for f in l:
    if os.path.exists(f):
      if os.path.exists(f[:-4]+"_2.mat"): test_list.append([f[:-4]+"_2.mat", f, 2])
      if os.path.exists(f[:-4]+"_3.mat"): test_list.append([f[:-4]+"_3.mat", f, 3])
      if os.path.exists(f[:-4]+"_4.mat"): test_list.append([f[:-4]+"_4.mat", f, 4])
print('>> test list count : ', len(test_list))

# placeholder
input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))

# network
shared_model = tf.make_template('shared_model', model)
output_tensor, weights = shared_model(input_tensor)
#output_tensor, weights = model(input_tensor)

# session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Restore variables from checkpoint in disk.
saver = tf.train.Saver(weights)
ckpt = tf.train.get_checkpoint_state('checkpoints')
if ckpt and ckpt.model_checkpoint_path:
  saver.restore(sess, ckpt.model_checkpoint_path)

print('\n>> testing results : ')
psnr_list = []
for i in range(len(test_list)):
  mat_dict = scipy.io.loadmat(test_list[i][0])
  input_y = None
  if "img_2" in mat_dict: input_y = mat_dict["img_2"]
  elif "img_3" in mat_dict: input_y = mat_dict["img_3"]
  elif "img_4" in mat_dict: input_y = mat_dict["img_4"]
  else: continue
  gt_y = scipy.io.loadmat(test_list[i][1])['img_raw']

  start_time = time.time()
  img_vdsr_y = sess.run([output_tensor], feed_dict={input_tensor: np.resize(input_y, (1, input_y.shape[0], input_y.shape[1], 1))})
  img_vdsr_y = np.resize(img_vdsr_y, (input_y.shape[0], input_y.shape[1]))
  stop_time = time.time()

  psnr_bicub = psnr(input_y, gt_y, test_list[i][2])
  psnr_vdsr = psnr(img_vdsr_y, gt_y, test_list[i][2])
  print("%s : bicubic %f, vdsr %f." % (test_list[i][0], psnr_bicub, psnr_vdsr))
  psnr_list.append([psnr_bicub, psnr_vdsr, test_list[i][2]])

  img = Image.fromarray((gt_y * 255).astype(np.uint8))
  img.save('./out/' + test_list[i][0].split('\\')[-1] + '_gt.bmp')
  img = Image.fromarray((input_y * 255).astype(np.uint8))
  img.save('./out/' + test_list[i][0].split('\\')[-1] + '_bicubic.bmp')
  img = Image.fromarray((img_vdsr_y * 255).astype(np.uint8))
  img.save('./out/' + test_list[i][0].split('\\')[-1] + '_vdsr.bmp')

sess.close()

with open('psnr_results.csv', 'w', newline='') as f:
  writer = csv.writer(f)
  column_names = ['bicubic']
  column_names.append('vdsr')
  column_names.append('scale')
  writer.writerow(column_names)
  writer.writerows(psnr_list)

print("Writing completed!!")

