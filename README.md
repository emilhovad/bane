# bane
Analysis of rail images and track recording car.

First the original repo is from https://github.com/YunYang1994/tensorflow-yolov3

NOTE, we choosed the folder name /tensorflow_yolov3 instead of /tensorflow-yolov3 (might be problematic)

If you want to train an example of a simple 1-class data set, a raccoon data set is provided and explained in the readme.md or look in the original repo...


#######
#Here it is for training the banedk images, 

Train "train image" dataset

To help you understand my training process, I made this training-pipline. The train dataset has three classes with alot of images, 


1) First you need to convert the original yolo label (many txt files in the folder /yolo_label/) to the labelling used in this tensorflow-yolov3 framework (1 txt file in the folder /tf_yolo_label/), remember to change the folders to where your images and yolo labelling are

$ Change_Labelling.py
This will put the new labelling (1 txt file) for this framework into the folder /tf_yolo_label/
This txt file will be in the format of,
#image_path x_min y_min x_max y_max class_id  x_min y_min ... class_id 
xxx/xxx.jpg 18.19 6.32 424.13 421.83 20 323.86 2.65 640.0 421.94 20 
xxx/xxx.jpg 55.38 132.63 519.84 380.4 16


2) Then a shell script is prepared in the ./scripts which enables you to get data and train it !
how to train it ? 

$ sh scripts/make_bane.sh # check folders in the bash script, but it should be ok. 
It should give you the output "Saving 1236 images in ./tf_yolo_label/bane_test.tfrecords", which means that two files are created a) bane_test.tfrecords and b) bane_train.tfrecords

3) Check the labelling in the tfrecords format

$ python show_input_image_bane.py          # show your input image (optional)


4) Make a anchor file anchor.txt and put in the /data/ folder based on his readme.md file (train your own data)

$ python kmeans_bane.py    # get prior anchors and rescale the values to the range [0,1]


5) Get the pretrained weights (downloads automatically if no files are in /checkpoint/) this is probably where you can choose (320X320, 416X1416 or 608X608) in the url download or if you put the weights in the /checkpoint/ folder 

$ python convert_weight_tf_bane.py --convert       # get pretrained weights
Remember our files are black white, we will convert the images to rgb and use three channels

6) Train your network, see in the top of the python file for the settings and make settings for our pc/cluster (it is somewhat intuitive)

$ python quick_train_bane.py

$ tensorboard --logdir ./data

As you can see in the tensorboard, if your dataset is too small or you train for too long, the model starts to overfit and learn patterns from training data that does not generalize to the test data.
how to test and evaluate it ?

7) Depending on what "if (epoch+1) % 800 == 0: saver.save(sess, save_path="./checkpoint/yolov3.ckpt", global_step=epoch+1))" saves with a multiplum of modulus = 800, this can be converted to a pb file which is what is used in tf

$ python convert_weight_tf_bane.py -cf ./checkpoint/yolov3.ckpt-800 -nc 3 -ap ./data/anchors.txt --freeze

Creates a './checkpoint/yolov3_cpu_nms.pb' file and a './checkpoint/yolov3_gpu_nms.pb' file.

8) Test the pb files (cpu and/or gpu), in this case it is 608X608X3 (3 color channels!)

$ python quick_test_bane_tf_trained.py

9) Todo get the evaluation code to work with our three class example, it works for the racoon example!
$ python evaluate_bane.py


#######
# Convert weight trained in the pjreddie framework to the tensorflow-yolov3 framework, 

1) use the code to convert the pjreddie trained network (darknet) to the this framework, this code convert the network with only 1 colour channel

$python convert_weight_darknet_Thomas.py --convert

=> model saved in path: ./checkpoint/yolov3darknet.ckpt (you should get this output)

2) convert yolov3darknet.ckpt to pb files 
when testing remember this network is 608X608X1 (1 color channel!)

#python convert_weight_darknet_Thomas.py -cf ./checkpoint/yolov3darknet.ckpt -nc 3 -ap ./data/anchors.txt --freeze

=> 1354 ops written to ./checkpoint/yolov3bane_cpu_nms.pb.
=> 1607 ops written to ./checkpoint/yolov3bane_gpu_nms.pb. (you should get this output)

3) Test the pb files (cpu and/or gpu), in this case it is 608X608X1 (1 color channels!)

#quick_test_bane.py (get a picture with a prediction)
