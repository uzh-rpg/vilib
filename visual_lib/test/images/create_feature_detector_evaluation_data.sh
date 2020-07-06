#!/bin/bash
echo "We are about to:"
echo " - download the EuRoC machine hall 1 dataset,"
echo " - extract the 752x480 images from the bag file, and"
echo " - also crop the images to 640x480"
echo "Make sure you have enough space on your disk ~ 4 GB!"
echo ""
read -p "[0] Are you sure you want to do this? [y/n] " ANSWER
if [ "${ANSWER}" != "y" ]; then
  exit 0
fi

TARGET_FOLDER="euroc"
BAG_NAME="MH_01_easy.bag"

DO_DATASET_INITIALIZATION=1
if [ ${DO_DATASET_INITIALIZATION} -eq 1 ]; then
  rm -rf ${TARGET_FOLDER}
  mkdir ${TARGET_FOLDER}
  cd ${TARGET_FOLDER}

  # Get dataset
  read -p " Do you have the dataset already? [y/n] " ANSWER
  if [ "${ANSWER}" != "y" ]; then
    wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.bag
  else
    read -p " Where? Provide full path to the bag file, please. [path] " ANSWER
    ln -s ${ANSWER} ${BAG_NAME}
  fi
else
  cd ${TARGET_FOLDER}
fi

echo "[1] Dataset ready"
read -p " Extract all images? [0 = all / N] " IMAGE_CNT_LIMIT
echo " Next: start extraction"
sleep 1

# At this point we have either a symlink or file in the folder
# Extract the images
EXTRACT_PYTHON_SCRIPT_NAME="extract_python.py"
cat << EOF > ${EXTRACT_PYTHON_SCRIPT_NAME}
#!/usr/bin/python
import rosbag
import argparse
import sys
import os
import cv2
from cv_bridge import CvBridge

parser = argparse.ArgumentParser(description='Extract left and right images from captured bags. The script upon request concatenates the left and right images for easier checking.')
parser.add_argument('bag', metavar='bag', help='ROS bag file to be analyzed')
parser.add_argument('output_dir', metavar='output_dir', help='Output directory')
parser.add_argument('image_cnt', metavar='image_cnt', help='Number of images to be extracted')
parsed = parser.parse_args();

if (os.path.isfile(parsed.bag) == False):
  print ('The specified file %s is not available' % parsed.bag)
  print ('Please select an existing bag file');
  sys.exit(-1);

if (os.path.exists(parsed.output_dir) == True):
  print ('The specified folder %s already exists' % parsed.output_dir);
  print ('Please select a not existing directory.')
  sys.exit(-1);
else:
  os.makedirs(parsed.output_dir);

bridge = CvBridge();
topic_to_extract = '/cam0/image_raw';
output_sub_dir = '752_480';
image_cnt_extracted = 0;
image_cnt_limit = int(parsed.image_cnt);

with rosbag.Bag(parsed.bag, 'r') as bag_in:
  # Save topics individually first
  if (os.path.exists(os.path.join(parsed.output_dir,output_sub_dir)) == False):
    os.makedirs(os.path.join(parsed.output_dir,output_sub_dir));
    img_base_path = os.path.join(parsed.output_dir,output_sub_dir);
  print ('Topic information: %s\t|\t%d' % (topic_to_extract,bag_in.get_message_count(topic_to_extract)));
  for topic, msg, t in bag_in.read_messages(topics=[topic_to_extract]):
    # extract timestamp
    nsec = (msg.header.stamp.secs*1e9) + msg.header.stamp.nsecs;
    # extract Image data
    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    cv2.imwrite(os.path.join(img_base_path, '%d.png' % nsec), cv_img)
    print ('%d.png' % nsec)
    image_cnt_extracted = image_cnt_extracted +1;
    if(image_cnt_extracted == image_cnt_limit):
      break;

EOF
chmod +x ${EXTRACT_PYTHON_SCRIPT_NAME}

DO_EXTRACTION=1
if [ ${DO_EXTRACTION} -eq 1 ]; then
  ./${EXTRACT_PYTHON_SCRIPT_NAME} ${BAG_NAME} ./images ${IMAGE_CNT_LIMIT}
fi

echo "[2] Extraction is ready"
echo " Next: start cropping to 640x480"
sleep 1

# do the cropping
DO_CROPPING=1
if [ ${DO_CROPPING} -eq 1 ]; then
  mkdir -p images/640_480
  cd images/752_480
  for file in *.png; do
    convert -crop 640x480+56+0 $file ../640_480/${file%.png}.png
    echo ${file}
  done
else
  cd images/752_480
fi

IMAGE_COUNT=$(ls -1 -v | wc -l)

echo "[3] Images are ready"
echo " Next: create list files for the test environment"
echo " Image count: "${IMAGE_COUNT}
sleep 1

cd ../..

IMAGE_LIST_FILE_752_480="image_list_752_480.txt"
IMAGE_LIST_FILE_640_480="image_list_640_480.txt"

echo -e "752\n480" > ${IMAGE_LIST_FILE_752_480}
ls -v -1 "images/752_480" | awk -v path="test/images/${TARGET_FOLDER}/images/752_480" '{print ""path"/"$1""}' >> ${IMAGE_LIST_FILE_752_480}
echo -e "640\n480" > ${IMAGE_LIST_FILE_640_480}
ls -v -1 "images/752_480" | awk -v path="test/images/${TARGET_FOLDER}/images/640_480" '{print ""path"/"$1""}' >> ${IMAGE_LIST_FILE_640_480}

echo "DONE"
