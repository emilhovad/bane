cat ./tf_yolo_label/labelThomas.txt | head -n  180 > ./tf_yolo_label/train.txt
cat ./tf_yolo_label/labelThomas.txt | tail -n +181 > ./tf_yolo_label/test.txt
python core/convert_tfrecord.py --dataset_txt ./tf_yolo_label/train.txt --tfrecord_path_prefix ./tf_yolo_label/bane_train
python core/convert_tfrecord.py --dataset_txt ./tf_yolo_label/test.txt  --tfrecord_path_prefix ./tf_yolo_label/bane_test
