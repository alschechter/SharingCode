import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import time
#%load_ext tensorboard 
#reload extension
from tensorflow.keras import layers
import seaborn as sns
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import sklearn
import glob
import re
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tqdm import tqdm 
from tensorflow.keras.regularizers import l2
from astropy.visualization import AsinhStretch
# import noise layer

shard_dir = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/z1mocks/sharding/'

for s in glob.glob(shard_dir + 'training/*.tfrecords'):
    os.remove(s)
for s in glob.glob(shard_dir + 'validation/*.tfrecords'):
    os.remove(s)
for s in glob.glob(shard_dir + 'test/*.tfrecords'):
    os.remove(s)
    
checkpoint_path = "/n/holystore01/LABS/hernquist_lab/Users/aschechter/checkpoints_z1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
tf.keras.backend.clear_session()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
        print(e)

epochs = 250
batch_size = 32
#batching helps computer memory challenges

data_augmentation1 = tf.keras.Sequential([keras.layers.RandomFlip("horizontal_and_vertical")])
data_augmentation2 = tf.keras.Sequential([keras.layers.RandomRotation(0.15)])
data_augmentation3 = tf.keras.Sequential([keras.layers.RandomZoom(height_factor = -0.2, width_factor = -0.2, fill_mode = 'constant' )])
data_aug_options = [data_augmentation1, data_augmentation2, data_augmentation3]

# parser = argparse.ArgumentParser()
# parser.add_argument(' -- epochs', type=int, default=100)
# parser.add_argument(' -- batch_size', type=int, default=64)
# parser.add_argument(' -- n_gpus', type=int, default=1)

# args = parser.parse_args()
# batch_size = args.batch_size
# epochs = args.epochs
# n_gpus = args.n_gpus

# device_type = 'GPU'
# devices = tf.config.experimental.list_physical_devices(
#           device_type)
# devices_names = [d.name.split("e:")[1] for d in devices]


path = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/z1mocks/'
train_mergers = []
train_mergers_labels = []
# train_earlymergers = []
# train_earlymergers_labels = []
# train_latemergers = []
# train_latemergers_labels = []
train_nonmergers = []
train_nonmergers_labels = []

val_mergers = []
val_mergers_labels = []
# val_earlymergers = []
# val_earlymergers_labels = []
# val_latemergers = []
# val_latemergers_labels = []
val_nonmergers = []
val_nonmergers_labels = []

test_mergers = []
test_mergers_labels = []
# test_earlymergers = []
# test_earlymergers_labels = []
# test_latemergers = []
# test_latemergers_labels = []
test_nonmergers = []
test_nonmergers_labels = []

stretch = AsinhStretch()

checkingnormalization = []
def normalize_image(image):
    image_zeromin = image - np.min(image)
    # if np.min(image) < 0: 
    #     image_zeromin = image + abs(np.min(image))
    # else:
    #     image_zeromin = image - abs(np.min(image))
    image_ZerotoOne = image_zeromin/(np.max(image_zeromin) - np.min(image_zeromin))
    if np.min(image_ZerotoOne) != 0:
        checkingnormalization.append('min not zero!')
    elif np.max(image_ZerotoOne) != 1:
        checkingnormalization.append('max not one!')
    elif np.min(image_ZerotoOne) == 0 and np.max(image_ZerotoOne) == 1:
        checkingnormalization.append('correct max and min!')
    return image_ZerotoOne


### images not to use ### 
sketchy_nonmergers = [361, 118818, 118843, 133864, 135875, 146554, 148683, 158279, 160380, 174083, 184810, 280800, 290860, 309092, 348113, 369391, 376748, 379695, 431980, 446009, 455980, 457553, 463603, 488076, 551201, 567964, 611832, 639298, 640580]

sketchy_nonmergers = list(map(str, sketchy_nonmergers))

for file in glob.glob(path + 'training/mergers/allfilters*.npy'):
    f = np.load(file)
    num_transformations_to_apply = np.random.randint(1, 4)
    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
    # choose a random transformation to apply for a single image
        key = np.random.choice(data_aug_options)
        f_aug = key(f)
        f_aug_norm = normalize_image(f_aug)
        f_aug_stretch = stretch(f_aug_norm)
        num_transformations += 1
        train_mergers.append(f_aug_stretch)
        label = 'merger'
        train_mergers_labels.append(label)
    
for file in glob.glob(path + 'training/earlymergers/allfilters*.npy'):
    f = np.load(file)
    #f = f/np.max(f)
    num_transformations_to_apply = np.random.randint(1, 4)
    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
        # choose a random transformation to apply for a single image
        key = np.random.choice(data_aug_options)
        f_aug = key(f)
        f_aug_norm = normalize_image(f_aug)
        f_aug_stretch = stretch(f_aug_norm)
        num_transformations += 1
        train_mergers.append(f_aug_stretch)
        label = 'merger'
        train_mergers_labels.append(label)
    
for file in glob.glob(path + 'training/latemergers/allfilters*.npy'):
    f = np.load(file)
    #f = f/np.max(f)
    num_transformations_to_apply = np.random.randint(1, 4)
    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
    # choose a random transformation to apply for a single image
        key = np.random.choice(data_aug_options)
        f_aug = key(f)
        f_aug_norm = normalize_image(f_aug)
        f_aug_stretch = stretch(f_aug_norm)
        num_transformations += 1
        train_mergers.append(f_aug_stretch)
        label = 'merger'
        train_mergers_labels.append(label)   

for file in glob.glob(path + 'training/nonmergers/allfilters*.npy'):
    subid = file[89:-6]
    if subid not in sketchy_nonmergers:
        f = np.load(file)
        #f = normalize_image(f)
        num_transformations_to_apply = np.random.randint(1, 4)
        num_transformations = 0
        transformed_image = None
        while num_transformations <= num_transformations_to_apply:
            # choose a random transformation to apply for a single image
            key = np.random.choice(data_aug_options)
            f_aug = key(f)
            f_aug_norm = normalize_image(f_aug)
            f_aug_stretch = stretch(f_aug_norm)
            num_transformations += 1
            train_nonmergers.append(f_aug_stretch)
            label = 'nonmerger'
            train_nonmergers_labels.append(label) 
    else:
        print('sketchy!!')
    
for file in glob.glob(path + 'validation/mergers/allfilters*.npy'):
    f = np.load(file)
    #f = normalize_image(f)
    num_transformations_to_apply = np.random.randint(1, 4)
    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
        # choose a random transformation to apply for a single image
        key = np.random.choice(data_aug_options)
        f_aug = key(f)
        f_aug_norm = normalize_image(f_aug)
        f_aug_stretch = stretch(f_aug_norm)
        num_transformations += 1
        val_mergers.append(f_aug_stretch)
        label = 'merger'
        val_mergers_labels.append(label)
    
for file in glob.glob(path + 'validation/earlymergers/allfilters*.npy'):
    f = np.load(file)
#    f = f/np.max(f)
    num_transformations_to_apply = np.random.randint(1, 4)
    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
        # choose a random transformation to apply for a single image
        key = np.random.choice(data_aug_options)
        f_aug = key(f)
        f_aug_norm = normalize_image(f_aug)
        f_aug_stretch = stretch(f_aug_norm)
        num_transformations += 1
        val_mergers.append(f_aug_stretch)
        label = 'merger'
        val_mergers_labels.append(label)
    
for file in glob.glob(path + 'validation/latemergers/allfilters*.npy'):
    f = np.load(file)
    #f = normalize_image(f)
    num_transformations_to_apply = np.random.randint(1, 4)
    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
        # choose a random transformation to apply for a single image
        key = np.random.choice(data_aug_options)
        f_aug = key(f)
        f_aug_norm = normalize_image(f_aug)
        f_aug_stretch = stretch(f_aug_norm)
        num_transformations += 1
        val_mergers.append(f_aug_stretch)
        label = 'merger'
        val_mergers_labels.append(label)   

for file in glob.glob(path + 'validation/nonmergers/allfilters*.npy'):
    subid = file[91:-6]
    if subid not in sketchy_nonmergers:
        f = np.load(file)
        #f = normalize_image(f)
        num_transformations_to_apply = np.random.randint(1, 4)
        num_transformations = 0
        transformed_image = None
        while num_transformations <= num_transformations_to_apply:
            # choose a random transformation to apply for a single image
            key = np.random.choice(data_aug_options)
            f_aug = key(f)
            f_aug_norm = normalize_image(f_aug)
            f_aug_stretch = stretch(f_aug_norm)
            num_transformations += 1
            val_nonmergers.append(f_aug_stretch)
            label = 'nonmerger'
            val_nonmergers_labels.append(label) 

    else:
        print('sketchy!!')

for file in glob.glob(path + 'test/mergers/allfilters*.npy'):
    f = np.load(file)
    #f = normalize_image(f)
    num_transformations_to_apply = np.random.randint(1, 4)
    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
        # choose a random transformation to apply for a single image
        key = np.random.choice(data_aug_options)
        f_aug = key(f)
        f_aug_norm = normalize_image(f_aug)
        f_aug_stretch = stretch(f_aug_norm)
        num_transformations += 1
        test_mergers.append(f_aug_stretch)
        label = 'merger'
        test_mergers_labels.append(label)

for file in glob.glob(path + 'test/earlymergers/allfilters*.npy'):
    f = np.load(file)
    #f = normalize_image(f)
    num_transformations_to_apply = np.random.randint(1, 4)
    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
        # choose a random transformation to apply for a single image
        key = np.random.choice(data_aug_options)
        f_aug = key(f)
        f_aug_norm = normalize_image(f_aug)
        f_aug_stretch = stretch(f_aug_norm)
        num_transformations += 1
        test_mergers.append(f_aug_stretch)
        label = 'merger'
        test_mergers_labels.append(label)
    
    
for file in glob.glob(path + 'test/latemergers/allfilters*.npy'):
    f = np.load(file)
    #f = normalize_image(f)
    num_transformations_to_apply = np.random.randint(1, 4)
    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
        # choose a random transformation to apply for a single image
        key = np.random.choice(data_aug_options)
        f_aug = key(f)
        f_aug_norm = normalize_image(f_aug)
        f_aug_stretch = stretch(f_aug_norm)
        num_transformations += 1
        test_mergers.append(f_aug_stretch)
        label = 'merger'
        test_mergers_labels.append(label)   

for file in glob.glob(path + 'test/nonmergers/allfilters*.npy'):
    subid = file[85:-6]
    if subid not in sketchy_nonmergers:
        f = np.load(file)
        #f = normalize_image(f)
        num_transformations_to_apply = np.random.randint(1, 4)
        num_transformations = 0
        transformed_image = None
        while num_transformations <= num_transformations_to_apply:
            # choose a random transformation to apply for a single image
            key = np.random.choice(data_aug_options)
            f_aug = key(f)
            f_aug_norm = normalize_image(f_aug)
            f_aug_stretch = stretch(f_aug_norm)
            num_transformations += 1
            test_nonmergers.append(f_aug_stretch)
            label = 'nonmerger'
            test_nonmergers_labels.append(label) 

    else:
        print('sketchy!!')


train_mergers = np.array(train_mergers)
train_mergers_labels = np.array(train_mergers_labels, dtype = str)
# train_earlymergers = np.array(train_earlymergers)
# train_earlymergers_labels = np.array(train_earlymergers_labels, dtype = str)
# train_latemergers = np.array(train_latemergers)
# train_latemergers_labels = np.array(train_latemergers_labels, dtype = str)
train_nonmergers = np.array(train_nonmergers)
train_nonmergers_labels = np.array(train_nonmergers_labels, dtype = str)

val_mergers = np.array(val_mergers)
val_mergers_labels = np.array(val_mergers_labels, dtype = str)
# val_earlymergers = np.array(val_earlymergers)
# val_earlymergers_labels = np.array(val_earlymergers_labels, dtype = str)
# val_latemergers = np.array(val_latemergers)
# val_latemergers_labels = np.array(val_latemergers_labels, dtype = str)
val_nonmergers = np.array(val_nonmergers)
val_nonmergers_labels = np.array(val_nonmergers_labels, dtype = str)

## for interpretation
#np.savetxt('output.txt', np.shape(val_mergers[0,:,:,:]))
#exit()
# interptestdata = np.concatenate((val_mergers[0,:,:,:], val_earlymergers[0,:,:,:], val_latemergers[0,:,:,:], val_nonmergers[0,:,:,:]))
# np.save('interpretationdata.npy', interptestdata)

test_mergers = np.array(test_mergers)
test_mergers_labels = np.array(test_mergers_labels, dtype = str)
# test_earlymergers = np.array(test_earlymergers)
# test_earlymergers_labels = np.array(test_earlymergers_labels, dtype = str)
# test_latemergers = np.array(test_latemergers)
# test_latemergers_labels = np.array(test_latemergers_labels, dtype = str)
test_nonmergers = np.array(test_nonmergers)
test_nonmergers_labels = np.array(test_nonmergers_labels, dtype = str)    
    
trainingdata = np.concatenate((train_mergers,train_nonmergers))
traininglabels = np.concatenate((train_mergers_labels, train_nonmergers_labels))
valdata = np.concatenate((val_mergers, val_nonmergers))
vallabels = np.concatenate((val_mergers_labels, val_nonmergers_labels))
testdata = np.concatenate((test_mergers, test_nonmergers))
testlabels = np.concatenate((test_mergers_labels, test_nonmergers_labels))

print('training shape: ', np.shape(trainingdata))
print(np.shape(trainingdata[0]))

labelencoder = LabelEncoder()
if type(traininglabels[0] == str):
    print('doesnt work')
    print(type(traininglabels[0]))
traininglabels = labelencoder.fit_transform(traininglabels)
print(type(traininglabels[0]))
#exit()
vallabels = labelencoder.fit_transform(vallabels)
testlabels = labelencoder.fit_transform(testlabels)
train_num = len(trainingdata)
valid_num = len(valdata)

print('made it to 335')
#np.savetxt('lenths.txt', len(trainingdata)+len(valdata))
#exit()
# trainingdata = np.resize(trainingdata, (1, 202, 202, 4))
# valdata = np.resize(valdata, (1, 202, 202, 4))
# testdata = np.resize(testdata, (1, 202, 202, 4))

# train_dataset = tf.data.Dataset.from_tensor_slices((trainingdata, traininglabels)).shuffle(len(trainingdata), reshuffle_each_iteration=True)
# validation_dataset = tf.data.Dataset.from_tensor_slices((valdata, vallabels)).shuffle(len(valdata), reshuffle_each_iteration=True)
# input_context = tf.distribute.InputContext(
#     input_pipeline_id=1,  # Worker id
#     num_input_pipelines=4,  # Total number of workers
# )
# read_config = tfds.ReadConfig(
#     input_context=input_context,
# )
# options = tf.data.Options()
# options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO


#data set is too big to use from_tensor_slices!! Need to use TFRecords
#train_dataset = tf.data.Dataset.from_tensor_slices((trainingdata, traininglabels))
#functions below from this article: https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c

# def _bytes_feature(value):
#     """Returns a bytes_list from a string / byte."""
#     if isinstance(value, type(tf.constant(0))): # if value ist tensor
#         value = value.numpy() # get value of tensor
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# def _float_feature(value):
#     """Returns a floast_list from a float / double."""
#     return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

# def _int64_feature(value):
#     """Returns an int64_list from a bool / enum / int / uint."""
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# def serialize_array(array):
#     array = tf.io.serialize_tensor(array)
#     return array
# def parse_single_image(image, label):
  
#     #define the dictionary -- the structure -- of our single example
#     data = {
#         'height' : _int64_feature(image.shape[0]),
#         'width' : _int64_feature(image.shape[1]),
#         'depth' : _int64_feature(image.shape[2]),
#         'raw_image' : _bytes_feature(serialize_array(image)),
#         'label' : _int64_feature(label)
#         }
#     #create an Example, wrapping the single features
#     out = tf.train.Example(features=tf.train.Features(feature=data))

#     return out
# def write_images_to_tfr_short(images, labels, filename="images"):
#     filename= filename+".tfrecords"
#     writer = tf.io.TFRecordWriter(filename) #create a writer that'll store our data to disk
#     count = 0

#     for index in range(len(images)):

#         #get the data we want to write
#         current_image = images[index] 
#         current_label = labels[index]

#         out = parse_single_image(image=current_image, label=current_label)
#         writer.write(out.SerializeToString())
#         count += 1

#     writer.close()
#     print(f"Wrote {count} elements to TFRecord")
#     return count


# def write_images_to_tfr_long(images, labels, filename="z1images", max_files=50, out_dir=shard_dir):

#     #determine the number of shards (single TFRecord files) we need:
#     splits = (len(images)//max_files) + 1 #determine how many tfr shards are needed
#     if len(images)%max_files == 0:
#         splits-=1
#     print(f"\nUsing {splits} shard(s) for {len(images)} files, with up to {max_files} samples per shard")

#     file_count = 0
    
#     for i in tqdm(range(splits)):
#         current_shard_name = "{}{}_{}{}.tfrecords".format(out_dir, i+1, splits, filename)
#         writer = tf.io.TFRecordWriter(current_shard_name)

#         current_shard_count = 0
#         while current_shard_count < max_files: #as long as our shard is not full
#             #get the index of the file that we want to parse now
#             index = i*max_files+current_shard_count
#             if index == len(images): #when we have consumed the whole data, preempt generation
#                 break
                
#             current_image = images[index]
#             current_label = labels[index]

#                   #create the required Example representation
#             out = parse_single_image(image=current_image, label=current_label)
    
#             writer.write(out.SerializeToString())
#             current_shard_count+=1
#             file_count += 1

#         writer.close()
#     print(f"\nWrote {file_count} elements to TFRecord")
#     return file_count

# ##Read back in!
# def parse_tfr_element(element):
#     #use the same structure as above; it's kinda an outline of the structure we now want to create
#     data = {
#         'height': tf.io.FixedLenFeature([], tf.int64),
#         'width':tf.io.FixedLenFeature([], tf.int64),
#         'label':tf.io.FixedLenFeature([], tf.int64),
#         'raw_image' : tf.io.FixedLenFeature([], tf.string),
#         'depth':tf.io.FixedLenFeature([], tf.int64),
#         }

    
#     content = tf.io.parse_single_example(element, data)
  
#     height = content['height']
#     width = content['width']
#     depth = content['depth']
#     label = content['label']
#     raw_image = content['raw_image']
  
  

#     feature = tf.io.parse_tensor(raw_image, out_type=tf.float32)
#     feature = tf.reshape(feature, shape=[height,width,depth])
#     return (feature, label)

# def get_dataset_large(tfr_dir = shard_dir, pattern="*z1images.tfrecords"):
#     files = glob.glob(tfr_dir+pattern, recursive=False)

#     #create the dataset
#     dataset = tf.data.TFRecordDataset(files)

#     #pass every single feature through our mapping function
#     dataset = dataset.map(
#         parse_tfr_element
#     )
    
#     return dataset


# trainingshards = write_images_to_tfr_long(images = trainingdata, labels = traininglabels, max_files=50, out_dir = shard_dir + '/training/')
# validationshards = write_images_to_tfr_long(images = valdata, labels = vallabels, max_files=50, out_dir = shard_dir + '/validation/')
# testshards = write_images_to_tfr_long(images = testdata, labels = testlabels, max_files=50, out_dir = shard_dir + '/test/')
# training_dataset = get_dataset_large()

# train_dataset = get_dataset_large(shard_dir + '/training/')
# validation_dataset = get_dataset_large(shard_dir + '/validation/')
# test_dataset = get_dataset_large(shard_dir + '/test/')
train_dataset = tf.data.Dataset.from_tensor_slices((trainingdata, traininglabels))
validation_dataset = tf.data.Dataset.from_tensor_slices((valdata, vallabels))
test_dataset = tf.data.Dataset.from_tensor_slices((testdata, testlabels))
#print(type(train_dataset))
print('made it to 495')
# for sample in train_dataset:
#     print(sample[0])
#     print(sample[1])
# exit()
# trainingdata = tf.train.FloatList(trainingdata)
# trainingdata = tf.train.Feature(float_list = trainingdata)
# traininglabels = bytes(traininglabels, 'utf-8')
# traininglabels = tf.train.Feature(bytes_list = traininglabels)
# print(traininglabels[0])
# exit()
# train_dataset = train_dataset.with_options(options)
# #train_dataset = train_dataset.shard(1,1)#num_workers, worker_index
train_dataset = train_dataset.repeat(epochs)
train_dataset = train_dataset.shuffle(len(trainingdata))
#train_dataset = train_dataset.map(map_func=self.parse_tfrecord, 
#                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(batch_size)
#train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# validation_dataset = tf.data.Dataset.from_tensor_slices((valdata, vallabels))
# validation_dataset = validation_dataset.with_options(options)
#validation_dataset = validation_dataset.shard(1,1)
validation_dataset = validation_dataset.repeat(epochs)
validation_dataset = validation_dataset.shuffle(len(valdata))
#validation_dataset = validation_dataset.map(map_func=self.parse_tfrecord, 
#                                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
validation_dataset = validation_dataset.batch(batch_size)
#validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# test_dataset = tf.data.Dataset.from_tensor_slices((testdata, testlabels))
# test_dataset = test_dataset.with_options(options)
# #test_dataset = test_dataset.shard(1,1)
test_dataset = test_dataset.repeat(epochs)
test_dataset = test_dataset.shuffle(len(testdata))
#test_dataset = test_dataset.map(map_func=self.parse_tfrecord, 
#                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size)
#test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#Sharding because now the dataset is really big! If I don't do this I run out of memory when running the network

    

print('elspec: ', validation_dataset.element_spec)
trainsavepath = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/z1mocks/train_tensors'
valsavepath = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/z1mocks/val_tensors'
#used to have some experimental save stuff here
#print(np.shape(train_dataset[0]))
#print(type(train_dataset))
#print(tf.shape(train_dataset))
# train_dataset = train_dataset.batch(32).repeat(epochs)
# validation_dataset = validation_dataset.batch(32).repeat(epochs)


img_rows, img_cols = 202, 202 #change to image dimensions
input_shape = (img_rows, img_cols, 4) #4 COSMOS filters

num_classes = 2

# data_augmentation = tf.keras.Sequential([
#   keras.layers.RandomFlip("horizontal_and_vertical"),
#   keras.layers.RandomRotation(0.15),
#   keras.layers.RandomZoom(height_factor = -0.1, width_factor = -0.1, fill_mode = 'constant' )
# ])
print('made it to the model')
#AlexNet architecture from towardsdatascience article --> changed to 4 image depth dimensions, change filter sizes to relative to 202 pixels
# model = keras.models.Sequential([
#     keras.layers.Conv2D(filters=64, kernel_size=(11,11), strides = (4,4),activation='relu', input_shape=(202,202,4), name = 'layer1'),
#     keras.layers.BatchNormalization(), #not sure if you should leave in or omit this line
#     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     keras.layers.GaussianNoise(stddev = 1),
#     keras.layers.Conv2D(filters=128, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same", name = 'layer2'),
#     # keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     # keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same", name = 'layer3'),
#     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     keras.layers.Flatten(),
#     keras.layers.Dense(2048, activation='relu'), #dense = fully connected
#     keras.layers.Dropout(0.2),
#     keras.layers.Dense(1024, activation='relu'),
#     keras.layers.Dropout(0.2),
#     keras.layers.Dense(2, activation='sigmoid')
# ])


#### 11/4 TRYING WITH WAY FEWER FILTERS ####
#try printing histograms of weights
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=8, kernel_size=(5,5),  strides=(2,2), activation='relu', input_shape=(202,202,4), padding="same", name = 'layer1'), #strides=(4,4),
    keras.layers.BatchNormalization(),
    #keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    #keras.layers.AveragePooling2D(pool_size=(3,3), strides=None, padding = 'valid'),
    #keras.layers.GaussianNoise(stddev = 1),
    keras.layers.Conv2D(filters=16, kernel_size=(4,4), strides=(2,2), activation='relu', padding="same", name = 'layer2'),
    keras.layers.BatchNormalization(),
    #keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    #keras.layers.AveragePooling2D(pool_size=(3,3), strides=None, padding = 'valid'),
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation='relu', padding="same", name = 'layer3'),
    #keras.layers.BatchNormalization(),
    #keras.layers.AveragePooling2D(pool_size=(3,3), strides=(2,2), padding = 'valid'),
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), activation='relu', padding="same", name = 'layer4'),
    keras.layers.BatchNormalization(),
    #keras.layers.AveragePooling2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), activation='relu', padding="same", name = 'layer5'),
    #keras.layers.BatchNormalization(),
    #keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    #keras.layers.AveragePooling2D(pool_size=(3,3), strides=(2,2), padding = "valid"),
    keras.layers.Flatten(),
    #keras.layers.Dense(4096, activation='relu'), #dense = fully connected
    keras.layers.Dense(256, kernel_regularizer=l2(0.0001), activity_regularizer=tf.keras.regularizers.L2(0.0001), activation = 'relu'),
    keras.layers.Dense(128, kernel_regularizer=l2(0.0001), activity_regularizer=tf.keras.regularizers.L2(0.0001), activation='relu'),
    keras.layers.Dense(64, kernel_regularizer=l2(0.0001), activity_regularizer=tf.keras.regularizers.L2(0.0001), activation='relu'), #dense = fully connected
    # keras.layers.Dense(4096),
    # keras.layers.LeakyReLU(alpha=0.3),
    keras.layers.Dropout(0.4),
    #keras.layers.Dense(4096, activation='relu'), #trying LeakrReLU
    keras.layers.Dense(32, kernel_regularizer=l2(0.0001), activity_regularizer=tf.keras.regularizers.L2(0.0001), activation='relu'), #dense = fully connected
    # keras.layers.Dense(4096),
    # keras.layers.LeakyReLU(alpha=0.3),
    keras.layers.Dropout(0.4), #get rid of this
    keras.layers.Dense(16, kernel_regularizer=l2(0.0001), activity_regularizer=tf.keras.regularizers.L2(0.0001), activation='relu'), #think really carefully about what's going on here! no dropout, no interesting activation functions!
    keras.layers.Dense(1, activation='sigmoid')
])


###
# model = keras.models.Sequential([
#     keras.layers.Conv2D(filters=96, kernel_size=(9,9),  strides=(2,2), activation='relu', input_shape=(202,202,4), padding="same", name = 'layer1'), #strides=(4,4),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     #keras.layers.GaussianNoise(stddev = 1),
#     keras.layers.Conv2D(filters=256, kernel_size=(4,4), strides=(2,2), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     # keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
#     # keras.layers.BatchNormalization(),
#     # keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
#     # keras.layers.BatchNormalization(),
#     # keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), activation='relu', padding="same", name = 'lastconvolution'),
#     # keras.layers.BatchNormalization(),
#     # keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     keras.layers.Flatten(),
#     #keras.layers.Dense(4096, activation='relu'), #dense = fully connected
#     keras.layers.Dense(2048, activation='relu'), #dense = fully connected
#     # keras.layers.Dense(4096),
#     # keras.layers.LeakyReLU(alpha=0.3),
#     keras.layers.Dropout(0.8),
#     #keras.layers.Dense(4096, activation='relu'), #trying LeakrReLU
#     keras.layers.Dense(1024, activation='relu'), #dense = fully connected
#     # keras.layers.Dense(4096),
#     # keras.layers.LeakyReLU(alpha=0.3),
#     keras.layers.Dropout(0.8),
#     keras.layers.Dense(512, activation='relu'), #I ADDED THIS LINE - NOT IN ALEX NET!!
#     keras.layers.Dropout(0.8),
#     keras.layers.Dense(2, activation='sigmoid')
# ])
# model = keras.models.Sequential([
#     data_augmentation,
#     keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(202,202,4)),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
#     keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
#     keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     keras.layers.Flatten(),
#     keras.layers.Dense(4096, activation='relu'), #dense = fully connected
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(4096, activation='relu'),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(4, activation='sigmoid')
# ])


tf.keras.utils.plot_model(model, to_file='Model_summary.png', show_shapes=True)
# some training parameters

mirrored_strategy = tf.distribute.MirroredStrategy()
replace2linear = ReplaceToLinear()

model.compile(loss=tf.keras.losses.binary_crossentropy, 
              optimizer=tf.optimizers.Adam(learning_rate=0.0005
              ), metrics=['accuracy']) #try with different optimizer
# score = CategoricalScore([0, 1, 2, 3])
# image_titles = ['Nonmerger', 'Merger', 'Early Stage Merger', 'Late Stage Merger']

print('compiled model')          
# with mirrored_strategy.scope():
#      model = model
#      opt = tf.keras.optimizers.Adam(0.001*n_gpus)
#      model.compile(loss='sparse_categorical_crossentropy' , 
#      optimizer=opt, metrics=['accuracy'])

tf.keras.utils.plot_model(model, to_file='Model_summary.png', show_shapes=True)


train_dir = '/n/home09/aschechter/code/z1mocks/training'
valid_dir = '/n/home09/aschechter/code/z1mocks/validation'

### DATA AUGMENTATION GOES HERE ###

root_logdir = os.path.join(os.curdir, "logs\\fit\\")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

 #metric was accuracy #can change optimizer if we want


### CALLBACKS AND CHECKPOINTS GO HERE ###
my_callbacks = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=2)
                                                 
csv_logger = tf.keras.callbacks.CSVLogger('training.log')

earlystopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    verbose=1,
    mode='auto',
    baseline=None,
    restore_best_weights=False
)

history = model.fit(train_dataset,
                    epochs=epochs,
                    steps_per_epoch=train_num // batch_size,
                    validation_data=validation_dataset,
                    validation_steps=valid_num // batch_size,
                    callbacks=[my_callbacks, csv_logger, earlystopping], #, earlystopping
                    verbose=2)

#look at order of magnitude of weights to troubleshoot 
for l in np.arange(1,6):
    layerweights = model.get_layer('layer' + str(l)).weights
    #np.save('BCweights/layer' + str(l) + 'weights.npy', layerweights)
    if l == 1:
        #layerweights = np.array(layerweights)
        print('data type = ', type(layerweights))
        #print('shape = ', np.shape(layerweights))
        print(layerweights[0])
        
        #np.savetxt('BCweights/layer' + str(l) + 'weights.txt', layerweights, dtype=object)
    
print('history keys: ', history.history.keys())
print(model.summary())
tf.keras.utils.plot_model(model, to_file='BCarchitecture.png', show_shapes=True)
model.save('BinaryModelz1_BC')

plt.figure()
plt.plot(history.history['accuracy'], label = 'train')
plt.plot(history.history['val_accuracy'], label = 'validation')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='upper left')
plt.savefig('BC_accuracy_binary_batch' + str(batch_size) + '_simple.png')
# summarize history for loss
plt.figure()
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label= 'validation')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper left')
plt.savefig('BC_loss_binary_batch' + str(batch_size) + '_simple.png')
#I think evaluation should go in a separate file and probably a jupyter notbeook

# testimage = np.load('/n/holystore01/LABS/hernquist_lab/Users/aschechter/z1mocks/validation/mergers/allfilters118813_1.npy')
# testimage = tf.convert_to_tensor(testimage)
# predictions = model.predict(testimage) #eventually replace with test
# N = len([testimage])
# y_predicted = np.zeros(N)
# for i in np.arange(0, N):
#     predictions_array = predictions[i,:]
#     predicted_label = np.argmax(predictions_array)
#     y_predicted[i] = int(predicted_label)
    
    
# y_actu = validation_dataset.astype(int)
# y_pred = y_predicted.astype(int)
# cm = confusion_matrix(y_actu, y_pred)
# categories = ['Merger', 'Early Stage', 'Late Stage', 'Nonmerger']
# group_names = ['True Neg','False Pos','False Neg','True Pos']
# group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
# labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names,group_percentages)]
# labels = np.asarray(labels).reshape(4,4)
# plt.figure()
# sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
# plt.savefig('ConfusionMatrix.png')



# # # Compute ROC curve and ROC area for each class
# # n_classes = 4
# # fpr = dict()
# # tpr = dict()
# # roc_auc = dict()
# # for i in range(n_classes):
# #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
# #     roc_auc[i] = auc(fpr[i], tpr[i])

# ### VISUALIZE FEATURE MAPS ###


# feature_extractor = keras.Model(
#     inputs=model.inputs,
#     outputs=model.get_layer(name="lastconvolution").output
# )
# # Call feature extractor on test input.
# #img2 = load_img('validation_z02/mergers/624423_z.png', target_size=(78, 78))
# features = feature_extractor.predict(testimage)
# fig = plt.figure(figsize = (10,10))
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
# for filt in range(np.shape(features)[-1]):
#     #i = i+500
#     ax = fig.add_subplot(16, 16, filt + 1, xticks=[], yticks=[])
#     ax.imshow(features[0,:,:,filt], cmap='plasma')
# fig.savefig('hiddenfeaturemaps.png')

# replace2linear = ReplaceToLinear()

# saliency = Saliency(model,
#                     model_modifier=replace2linear,
#                     clone=True)
# saliency_map = saliency(score, testimage)
# # Render
# ax1, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
# ax1.set_title('Merger', fontsize=16)
# ax1.imshow(testimage, cmap='plasma')
# #ax[i].imshow(image[i])
# ax1.axis('off')
# ax2.set_title('Saliency', fontsize=16)
# ax2.imshow(saliency_map, cmap='plasma')
# ax2.axis('off')
# plt.tight_layout()
# plt.savefig('Saliencymaps.png', dpi = 300)


