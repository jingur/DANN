wget -O uda_target_mnistm.pth https://www.dropbox.com/s/49sngfcsu2lnz9y/uda_target_mnistm.pth?dl=1
wget -O uda_target_svhn.pth https://www.dropbox.com/s/4ohowboad09x06a/uda_target_svhn.pth?dl=1
wget -O uda_target_usps.pth https://www.dropbox.com/s/2edflr36vyp1utl/uda_target_usps.pth?dl=1
python3 dann_test.py --data_dir $1 --target $2 --pred_path $3 --resume_mnistm uda_target_mnistm.pth --resume_svhn uda_target_svhn.pth --resume_usps uda_target_usps.pth --model_type uda