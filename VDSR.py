import glob
import numpy as np
import os
import random
import re
import scipy.io
import threading
import tensorflow as tf

from MODEL import model

DATA_PATH = "./data/train/"
IMG_SIZE = (41, 41)
BATCH_SIZE = 512
BASE_LR = 0.0001
LR_RATE = 0.1
LR_STEP_SIZE = 120
MAX_EPOCH = 120
USE_QUEUE_LOADING = True

# filelist
l = glob.glob(os.path.join(DATA_PATH, "*"))
print("total samples : ", len(l))
l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
print("total images : ", len(l))
train_list = []
for f in l:
  if os.path.exists(f):
    if os.path.exists(f[:-4]+"_2.mat"): train_list.append([f, f[:-4]+"_2.mat"])
    if os.path.exists(f[:-4]+"_3.mat"): train_list.append([f, f[:-4]+"_3.mat"])
    if os.path.exists(f[:-4]+"_4.mat"): train_list.append([f, f[:-4]+"_4.mat"])
random.shuffle(train_list)

# placeholder
train_input_single = tf.placeholder(tf.float32, shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
train_gt_single = tf.placeholder(tf.float32, shape=(IMG_SIZE[0], IMG_SIZE[1], 1))

# asynchronous queue loading
q = tf.FIFOQueue(10000, [tf.float32, tf.float32], [[IMG_SIZE[0], IMG_SIZE[1], 1], [IMG_SIZE[0], IMG_SIZE[1], 1]])
enqueue_op = q.enqueue([train_input_single, train_gt_single])
train_input, train_gt = q.dequeue_many(BATCH_SIZE)

# network
shared_model = tf.make_template('shared_model', model)
train_output, weights = shared_model(train_input)
#train_output, weights = model(train_input)

# loss
loss = tf.reduce_sum(tf.nn.l2_loss(tf.subtract(train_output, train_gt)))
for w in weights:
  loss += tf.nn.l2_loss(w)*1e-4
tf.summary.scalar("loss", loss)

# lr
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(BASE_LR, global_step*BATCH_SIZE, len(train_list)*LR_STEP_SIZE, LR_RATE, staircase=True)
tf.summary.scalar("learning_rate", learning_rate)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
opt = optimizer.minimize(loss, global_step=global_step)

# session
sess = tf.Session(config=tf.ConfigProto())
sess.run(tf.global_variables_initializer())

#TensorBoard open log with "tensorboard --logdir=logs"
if not os.path.exists('logs'):
  os.mkdir('logs')
merged = tf.summary.merge_all()
file_writer = tf.summary.FileWriter('logs', sess.graph)

# Restore variables from checkpoint in disk.
saver = tf.train.Saver(weights, max_to_keep=0)
ckpt = tf.train.get_checkpoint_state('checkpoints')
if ckpt and ckpt.model_checkpoint_path:
  saver.restore(sess, ckpt.model_checkpoint_path)
  epoch = ckpt.model_checkpoint_path.split('/')[-1].split('_')[-1].split('.')[0]
  if not epoch.isdigit():
    epoch = 0
  else:
    epoch = int(epoch)
else:
  tf.logging.info('No checkpoint file found, training debut...')
  epoch = 0

def load_and_enqueue(coord, file_list, enqueue_op, train_input_single, train_gt_single, idx=0, num_thread=1):
  count = 0;
  length = len(file_list)
  try:
    while not coord.should_stop():
      i = count % length;
      input_img = scipy.io.loadmat(file_list[i][1])['patch'].reshape([IMG_SIZE[0], IMG_SIZE[1], 1])
      gt_img = scipy.io.loadmat(file_list[i][0])['patch'].reshape([IMG_SIZE[0], IMG_SIZE[1], 1])
      sess.run(enqueue_op, feed_dict={train_input_single:input_img, train_gt_single:gt_img})
      count+=1
  except Exception as e:
    print("stopping...", idx, e)

# create threads
threads = []
num_thread=200
coord = tf.train.Coordinator()
for i in range(num_thread):
  length = int(len(train_list)/num_thread)
  t = threading.Thread(target=load_and_enqueue, args=(coord, train_list[i*length:(i+1)*length],enqueue_op, train_input_single, train_gt_single,  i, num_thread))
  threads.append(t)
  t.start()
print("threads number : " , len(threads))

try:
  while not coord.should_stop():
    max_step=len(train_list)//BATCH_SIZE
    for step in range(max_step):
      _,l,output,lr, g_step, summary = sess.run([opt, loss, train_output, learning_rate, global_step, merged])
      print("[epoch %2.4f] loss %.4f\t lr %.5f"%(epoch+(float(step)*BATCH_SIZE/len(train_list)), np.sum(l)/BATCH_SIZE, lr))
      file_writer.add_summary(summary, step+epoch*max_step)
      #print "[epoch %2.4f] loss %.4f\t lr %.5f\t norm %.2f"%(epoch+(float(step)*BATCH_SIZE/len(train_list)), np.sum(l)/BATCH_SIZE, lr, norm)
    epoch += 1
    saver.save(sess, "./checkpoints/VDSR_adam_epoch_%03d.ckpt" % epoch ,global_step=global_step)
except tf.errors.OutOfRangeError:
  print('Done training for %d epochs, %d steps.' % epoch)
finally:
  # When done, ask the threads to stop.
  coord.request_stop()

# Wait for threads to finish.
coord.join(threads)
sess.close()


#    print("not use queue loading, just sequential loading...")
#    ### WITHOUT ASYNCHRONOUS DATA LOADING ###
#    train_input = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1))
#    train_gt = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1))
#    ### WITHOUT ASYNCHRONOUS DATA LOADING ###


#def get_image_batch(train_list,offset,batch_size):
#  target_list = train_list[offset:offset+batch_size]
#  input_list = []
#  gt_list = []
#  cbcr_list = []
#  for pair in target_list:
#    input_img = scipy.io.loadmat(pair[1])['patch']
#    gt_img = scipy.io.loadmat(pair[0])['patch']
#    input_list.append(input_img)
#    gt_list.append(gt_img)
#  input_list = np.array(input_list)
#  input_list.resize([BATCH_SIZE, IMG_SIZE[1], IMG_SIZE[0], 1])
#  gt_list = np.array(gt_list)
#  gt_list.resize([BATCH_SIZE, IMG_SIZE[1], IMG_SIZE[0], 1])
#  return input_list, gt_list, np.array(cbcr_list)


#        for step in range(len(train_list)//BATCH_SIZE):
#          offset = step*BATCH_SIZE
#          input_data, gt_data, cbcr_data = get_image_batch(train_list, offset, BATCH_SIZE)
#          feed_dict = {train_input: input_data, train_gt: gt_data}
#          _,l,output,lr, g_step = sess.run([opt, loss, train_output, learning_rate, global_step], feed_dict=feed_dict)
#          print("[epoch %2.4f] loss %.4f\t lr %.5f"%(epoch+(float(step)*BATCH_SIZE/len(train_list)), np.sum(l)/BATCH_SIZE, lr))
#          del input_data, gt_data, cbcr_data
#        saver.save(sess, "./checkpoints/VDSR_const_clip_0.01_epoch_%03d.ckpt" % epoch ,global_step=global_step)

