python3 main.py --source_domain usps --target_domain mnistm --training_mode target --max_epoch 5 --fig_mode save | tee -a log-file
python3 main.py --source_domain mnistm --target_domain svhn --training_mode target --max_epoch 5 --fig_mode save | tee -a log-file
python3 main.py --source_domain svhn --target_domain usps --training_mode source --max_epoch 5 --fig_mode save | tee -a log-file

