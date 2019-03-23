from config import Configurations
import os
#import matplotlib.pyplot as plt
import skimage
import numpy as np
import keras 
import keras.layers as KL
import keras.models as KM
import keras.backend as K
import tensorflow as tf
import datetime
############################################################
#  Data Generator
############################################################
def data_preparation(path):
    '''
    input:
        path: path to the dataset
    return:
        image_info: infomation of the dataset as a list of dictionaries, eg:
        {'image_name': 'cat.11594.jpg',
        'class': 0,
        'path': '/home/yang/Downloads/dogs_vs_cats/dataset/validation/cats/cat.11594.jpg',
        'shape': (374, 500, 3)}
    note: class 0 for cat, 1 for dog
    '''
    
    # collect all the file names in the dirctory
    filenames = []
    for dirpath, dirnames, filenames_each_dir in os.walk(path):
        filenames.extend(filenames_each_dir)
    # cellect info of each image
    image_info = []
    for id, name in enumerate(filenames):
        if 'cat' in name:       
            image_path = os.path.join(path, 'cats', name)
            #img = skimage.io.imread(image_path)
            #image_shape = img.shape
            
            image_info.append({'image_name':name, 
                        'class':0,
                        'path':image_path,
                        #'shape':image_shape#
                        })
            
        elif 'dog' in name:
            image_path = os.path.join(path, 'dogs', name)
            #img = skimage.io.imread(image_path)
            #image_shape = img.shape
            image_info.append({'image_name':name, 
                        'class':1,
                        'path':image_path,
                        #'shape':image_shape#
                        })    
    return image_info

def data_generator(path, config):
    '''
    this function works as a generator
    input:
        data_path: path to the dataset
    return:
        a batch of images and labels
    '''
 
    image_info = data_preparation(path)
    # TODO: try to resize in a smarter way instead of fixed size
    '''
    get max H and W for resize image
    Max_H = Max_W = 0
    for index in range(len(image_info)):
        dictionary = image_info[index]
        H = dictionary['shape'][0]
        W = dictionary['shape'][1]
        if H > Max_H:
            Max_H = H
        if W > Max_W:
            Max_W = W
    shape_after_resize =  [Max_H, Max_W, 3]   
    '''

    # generator 
    # batch_index loops every batch and image_index loops every epoch
    batch_index = 0
    image_index = -1
    while True:
        try:
            image_index = (image_index + 1) % len(image_info)
            if image_index == 0:
                np.random.shuffle(image_info)
            # load image and class
            image_path = image_info[image_index]['path']
            image = skimage.io.imread(image_path)
            image_resized = skimage.transform.resize(image, config.IMAGE_SHAPE_AFTER_RESIZE)

            class_id = image_info[image_index]['class']
            # creat batch placeholders
            if batch_index == 0:
                batch_image = np.zeros((config.BATCH_SIZE,) + tuple(config.IMAGE_SHAPE_AFTER_RESIZE), dtype=image_resized.dtype)
                batch_class = np.zeros(config.BATCH_SIZE, dtype=np.int32)
            # add to batch
            batch_image[batch_index] = image_resized
            batch_class[batch_index] = class_id
                
            batch_index += 1
            
            if batch_index >= config.BATCH_SIZE:
                
                # note: returned varibles need to be correspond to inputs and outputs
                # here inputs = [batch_image, batch_class], outputs = []
                yield [batch_image, batch_class], []
                
                # start a new batch
                batch_index = 0
            
        except (GeneratorExit, KeyboardInterrupt):
            raise

############################################################
#  Dogsandcats class
############################################################
def smooth_l1_loss_graph(y_true, y_pred):
    '''
    inputs:
        y_true: [n, 1]
        y_pred: [n, 1]
    '''
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return K.mean(loss, axis=1)

def class_loss_graph(y_true, y_pred):
    '''
    inputs:
        y_true: [n, 1]
        y_pred: [n, 1]
    '''
    loss = K.binary_crossentropy(y_true, y_pred)
    return loss

class Dogsandcats(object):
    
    def __init__(self, mode, config):
        self.config = config
        self.mode = mode
        self.root_dir = os.path.abspath('.')
        self.model = self.build()
        self.set_log_dir()
        
    def build(self):
        # Inputs
        input_image = KL.Input(shape=self.config.INPUT_IMAGE_SHAPE, name='input_image')
        
        if self.mode == 'train':
            input_gt_class = KL.Input(shape=(1,), name='input_gt_class')
            
        x = KL.Conv2D(32, (3, 3), name='conv2d_1')(input_image)
        x = KL.BatchNormalization()(x)
        x = KL.Activation('relu')(x)
        x = KL.MaxPooling2D((2, 2), name='maxpooling_1')(x)
        
        x = KL.Conv2D(64, (3, 3), activation='relu', name='conv2d_2')(x)
        x = KL.BatchNormalization()(x)
        x = KL.Activation('relu')(x)
        x = KL.MaxPooling2D((2, 2), name='maxpooling_2')(x)      
        
        x = KL.Conv2D(128, (3, 3), activation='relu', name='conv2d_3')(x)
        x = KL.BatchNormalization()(x)
        x = KL.Activation('relu')(x)
        x = KL.MaxPooling2D((2, 2), name='maxpooling_3')(x) 
        
        x = KL.Flatten(name='flatten')(x)
        # choose sigmoid for activation
        output_prob = KL.Dense(1, activation='sigmoid', name='dense')(x)
        
        if self.mode == 'train':
            class_loss = KL.Lambda(lambda x: smooth_l1_loss_graph(*x), name='class_loss')([output_prob, input_gt_class])
            
            inputs = [input_image, input_gt_class]
            outputs = [output_prob, class_loss]             
        else:
            inputs = input_image
            outputs = output_prob
            
        model = KM.Model(inputs, outputs)
        model.summary()
        return model
    
    def train(self, train_generator, val_generator):
        # add loss
        # clear pervious losses
        self.model._losses = []
        self.model._per_input_losses = {}
        
        layer = self.model.get_layer('class_loss')
        #self.model.add_loss(layer.output)
        loss = (tf.reduce_mean(layer.output, keep_dims=True))
        self.model.add_loss(loss)        
        
        # compile the model
        optimizer = keras.optimizers.SGD(lr=self.config.LEARNING_RATE,
                                               momentum=self.config.MOMENTUM)
        # loss should be set to [None] * number of inputs
        self.model.compile(optimizer=optimizer,
                           loss=[None] * len(self.model.outputs))
        # add metrics for losses
        self.model.metrics_names.append('class_loss')
        self.model.metrics_tensors.append(loss)

        # create a checkpoint_dir if not exist
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        # callbacks    
        callbacks = [keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True)]        
        self.model.fit_generator(
            train_generator,
            epochs = self.config.EPOCHS,
            steps_per_epoch = self.config.STEPS_PER_EPOCH,
            validation_data = next(val_generator),
            validation_steps = self.config.VALIDATION_STEPS,
            callbacks = callbacks)

    def load_weights(self, model_path, by_name=True):
        '''
        load weights from trained model for detection
        inputs:
            path to model
        '''        
        import h5py
        from keras.engine import saving
        
        with h5py.File(model_path, 'r') as f:
            saving.load_weights_from_hdf5_group_by_name(f, self.model.layers)
    
    def evaluate(self, test_data_path, limit=0):
        '''
        evaluate the model with provide data
        inputs:
            path to test data
        returns:
            evaluation results test_image_info eg:{'image_id': 1429, 'name': '1429.jpg', 'image_path': '/home/yang/Downloads/dogs_vs_cats/dataset/test/1429.jpg', 'class': 0}
        '''
        # get image infor as a list of dictionaries, eg:{'image_id': 4124, 'name': '4124.jpg', 'image_path': '/home/yang/Downloads/dogs_vs_cats/dataset/test/4124.jpg', 'class': 1}
        
        test_image_info = []
        file_names = list(os.walk(test_data_path))[0][2]
            
        for name in file_names:
            image_id = int(name[:-4])
            image_path = os.path.join(test_data_path, name)
            test_image_info.append({'image_id':image_id, 
                                    'name':name, 
                                    'image_path':image_path})
        # sort the list in ascending order w.r.t. image_id
        test_image_info = sorted(test_image_info, key=lambda x:x['image_id'])  
        
        # apply limit
        if limit:
            test_image_info = test_image_info[:limit]
            
        # make prediction for each image and append results to test_image_info 
        for i, info in enumerate(test_image_info):
            # load image and resize
            image = skimage.io.imread(info['image_path'])
            image_resized = skimage.transform.resize(image, self.config.IMAGE_SHAPE_AFTER_RESIZE)
            # instead of [H, W, 3] imput image need to be [1, H, W, 3]
            molded_image_resized = image_resized[np.newaxis, :]

            output = self.model.predict([molded_image_resized])
            result = 0 if output[0][0] <=0.5 else 1
            print('Proccessing image {:>10}, it is a {}'.format(info['name'], 'cat' if result == 0 else 'dog'))
            test_image_info[i]['class'] = result
        
        for i, dic in enumerate(test_image_info):
            image_id = dic['image_id']
            
        return test_image_info

    def document_predictions(self, results):
        '''
        writh prediction to file
        '''
        # set the prediction file name
        now = datetime.datetime.now()
        filename = 'Predictions_{:%Y%m%dT%H%M}.csv'.format(now)
        file_path = os.path.join(self.log_dir, filename)
        
        # write to file
        print('Writing predictions to {}'.format(file_path))
        with open(file_path, 'w') as f:
            f.write('{:^30}\t{:^30}\n'.format('id', 'label'))
            
            for re in results:
                f.write('{:^30d}\t{:^30d}\n'.format(re['image_id'], re['class']))
    
    def set_log_dir(self):
        '''
        set the log_dir for the model to store checkpoint files
        '''
        # start a new training with epoch = 0
        #self.epoch = 0
        
        # set log path for all the logs
        self.log_dir = os.path.join(self.root_dir, 'logs')
        
        # set checkpoint path for each training
        model_name = self.config.MODEL_NAME
        now = datetime.datetime.now()
        self.checkpoint_dir = os.path.join(self.log_dir, '{}_{:%Y%m%dT%H%M}'.format(model_name.lower(), now))
        
        # set checkpoint file for each epoch, using palceholder *epoch*, keras will fill it with current epoch index                              
        self.checkpoint_path = os.path.join(self.checkpoint_dir, '{}_*epoch*.h5'.format(model_name.lower()))
        self.checkpoint_path = self.checkpoint_path.replace('*epoch*', '{epoch:04d}')
    
    
    
    
    
    
    
    
    
    
    
    