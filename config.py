"""
this is the configration file of dogs and cats classifier
"""
class Configurations(object):
    # model name
    MODEL_NAME = 'dogs_and_cats'
    
    # learing rate
    LEARNING_RATE = 1E-4
    
    # momentum of learning
    MOMENTUM = 0.9
    
    # note: this value should equals the shape of your post-resized images
    INPUT_IMAGE_SHAPE = [500, 500, 3]
    
    # how many epochs to train
    EPOCHS = 100
    
    # batch size
    BATCH_SIZE = 15
    
    # how many step for each epoch
    STEPS_PER_EPOCH = 20000 / BATCH_SIZE
    
    # steps for validation
    VALIDATION_STEPS = 5
    
    # shape after resize
    IMAGE_SHAPE_AFTER_RESIZE = INPUT_IMAGE_SHAPE
    
    #def __init__(self):
        #self.BATCH_SIZE = 20
        
    def display(self):
        print('\nConfigurations:')
        for attr in dir(self):
            if not attr.startswith('__') and \
            not callable(getattr(self, attr)):
                print('{:30}{}'.format(attr, getattr(self, attr)))
