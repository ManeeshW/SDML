grep REAL_IMG_DIR config/config.ini
$
docker run -it --rm --gpus all --ipc=host -v /home/maneesh/Desktop/LAB2.0/DATA-FDCL/Real_Data/examples/E_Test_5_2023.06.26/:/home/maneesh/Desktop/LAB2.0/DATA-FDCL/Real_Data/examples/E_Test_5_2023.06.26/ pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime



