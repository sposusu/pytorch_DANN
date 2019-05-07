import os, shutil

# dataset = 'mnistm'
dataset = 'svhn'
# dataset = 'usps'


data_dir = '../hw3-sposusu/hw3_data/digits/'+ dataset
train_labels = '../hw3-sposusu/hw3_data/digits/'+ dataset + '/train.csv'
test_labels = '../hw3-sposusu/hw3_data/digits/'+ dataset + '/test.csv'
train_images = '../hw3-sposusu/hw3_data/digits/'+ dataset + '/train'
test_images = '../hw3-sposusu/hw3_data/digits/'+ dataset + '/test'

def mkdirs(path):
    train_dir = path + '/' + 'train'
    test_dir = path + '/' + 'test'
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    for i in range(0, 10):
        if not os.path.exists(train_dir + '/' + str(i)):
            os.mkdir(train_dir + '/' + str(i))
        if not os.path.exists(test_dir + '/' + str(i)):
            os.mkdir(test_dir + '/' + str(i))

def process(labels_path, images_path, data_dir):
    with open(labels_path) as f:
        lines = iter(f.readlines())
        next(lines)
        for line in lines:
            line = line.replace('\n','')
            img = images_path + '/' + line.split(',')[0]
            dir = data_dir + '/' + line.split(',')[1] + '/' + line.split(',')[0]
            shutil.copyfile(img, dir)

mkdirs(data_dir)
process(train_labels, train_images, data_dir + '/train')
process(test_labels, test_images, data_dir + '/test')
# os.remove(train_images)
# os.remove(test_images)