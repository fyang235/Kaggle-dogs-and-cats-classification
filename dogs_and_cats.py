'''
usage:
    training:
        python dogs_and_cats.py mode --dataset=/path/to/dataset
        
        eg:
            python dogs_and_cats.py train --dataset=/home/yang/Downloads/dogs_vs_cats/dataset 
    
    evaluating:
        python dogs_and_cats.py evaluate --dataset=/path/to/dataset --model_path=/path/to/model
        
        eg:
            python dogs_and_cats.py evaluate --dataset=/home/yang/Downloads/dogs_vs_cats/dataset --model_path=/home/yang/deeplearning/dogs_and_cats/logs/dogs_and_cats_20190308T1002/dogs_and_cats_0002.h5
            
'''

####################################################################
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
print('############## GPU PREPARATION DONE ##############')
####################################################################


from model import Dogsandcats, data_generator
from config import Configurations
import os
PWD_dir = os.path.abspath('.')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='train a classifier on dogs and cats dataset.',
                                   )
    parser.add_argument('mode',
                       help='train or evaluate')
    parser.add_argument('--dataset',
                        required=True,
                        metavar='</path/to/data>',
                       help='set the path of dataset')
    parser.add_argument('--model_path',
                        metavar='</path/to/model>',
                        default=PWD_dir,
                       help='set the path of model')    
    args = parser.parse_args()
    
    data_path = args.dataset
    mode = args.mode
    model_path = args.model_path
    assert mode in ['train', 'evaluate']
    
    print('\n')
    print('{:30}{}'.format('mode: ', mode))
    print('{:30}{}'.format('dataset: ', data_path))
    print('{:30}{}'.format('model_path: ', model_path))
    
    # configure
    config = Configurations()
    
    # for traning
    if mode == 'train':
        # training data
        train_data_path = os.path.join(data_path, 'train')
        train_generator = data_generator(train_data_path, config)
        # validation data
        val_data_path = os.path.join(data_path, 'validation')
        val_generator = data_generator(val_data_path, config)
        
        # creat model
        model = Dogsandcats(mode, config)
        
        # train
        model.train(train_generator, val_generator)
        
    # for evaluatint
    else:
        # test data
        test_data_path = os.path.join(data_path, 'test')
        
        # creat model
        model = Dogsandcats(mode, config)
        model.load_weights(model_path)
        
        # test limit = 0 means test the entire dataset
        results = model.evaluate(test_data_path, limit=20)
        
        model.document_predictions(results)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    