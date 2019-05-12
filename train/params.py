from models import models

# utility params
fig_mode = None
embed_plot_epoch=10

# model params
use_gpu = True
dataset_mean = (0.5, 0.5, 0.5)
dataset_std = (0.5, 0.5, 0.5)

batch_size = 512
epochs = 1000
gamma = 10
theta = 1

# path params
data_root = '../hw3-sposusu/hw3_data/digits'

mnist_path = data_root + '/MNIST'
usps_path = data_root + '/usps'

mnistm_path = data_root + '/mnistm'
svhn_path = data_root + '/svhn'
syndig_path = data_root + '/SynthDigits'

save_dir = './experiment'


# specific dataset params
extractor_dict = {
                    'usps_mnistm': models.SVHN_Extractor(),
                  'mnistm_svhn': models.SVHN_Extractor(),
                  'svhn_usps': models.SVHN_Extractor(),
                  'mnistm':'feature_mnistm_10',
                  'svhn':"feature_svhn_10",
                  'usps':"feature_usps_10"
                  
                  }

class_dict = {
                'usps_mnistm': models.SVHN_Class_classifier(),
              'mnistm_svhn': models.SVHN_Class_classifier(),
              'svhn_usps': models.SVHN_Class_classifier(),
              'mnistm':'class_mnistm_10',
                  'svhn':"class_svhn_5",
                  'usps':"class_usps_10"}

domain_dict = {
                'usps_mnistm': models.SVHN_Domain_classifier(),
               'mnistm_svhn': models.SVHN_Domain_classifier(),
               'svhn_usps': models.SVHN_Domain_classifier(),
              'mnistm':'domainmnistm_10',
                  'svhn':"domainsvhn_5",
                  'usps':"domainusps_10"}
