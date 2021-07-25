from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import random
from time import sleep
from skimage import transform as trans
import cv2
import imageio
import pickle

from config.config import *
from face_detection import detect_face, face_image, face_preprocess


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--db',
                        default='db_demo',
                        type=str, help='Database name.')

    parser.add_argument('--create',
                        default=False,
                        type=bool, help='Set to True to create database if it does not exist.')

    parser.add_argument('--indir',
                        default='input/db_demo_images',
                        type=str, help='Directory with unaligned images.')

    parser.add_argument('--input-text',
                        default='input/db_demo_info.txt',
                        type=str, help='File to input information of known persons')

    parser.add_argument('--image-size', type=str, help='Image size (height, width) in pixels.', default='112,112')

    parser.add_argument('--exts', nargs='+', default=['.jpeg', '.jpg', '.png', '.bmp'],
                        help='list of acceptable image extensions.')

    parser.add_argument('--encoding',
                        default='utf-8',
                        help='Encoding of text files.')

    return parser.parse_args(argv)


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def main(args):
    # initialization
    database_dir = os.path.join(db_root_dir, args.db)
    if not os.path.exists(database_dir):
        if args.create:
            os.makedirs(database_dir)
        else:
            print('ERROR: Database "%s" not created. Abort.' % args.db)
            print('\tNote: To create a database, add "--create=True" before running the programme.')
            exit(0)
    thumb_dir = os.path.join(database_dir, db_thumb_dir)
    person_info_path = os.path.join(database_dir, db_person_info)
    data_bin_path = os.path.join(database_dir, db_data_bin)
    if not os.path.exists(args.indir):
        print('ERROR: Input directory "%s" not exist. Abort.' % args.indir)
        exit(0)
    if not os.path.exists(thumb_dir):
        os.makedirs(thumb_dir)
    if not os.path.exists(args.input_text):
        print('ERROR: Person information input file "%s" not exist. Abort.' % args.input_text)
        exit(0)
    elif os.path.normcase(args.input_text) == os.path.normcase(person_info_path):
        print('ERROR: Person information inputted from system database. Abort.')
        exit(0)

    # select valid inputs
    person_info_list_tmp = []
    person_no_list = []
    with open(args.input_text, 'r', encoding=args.encoding) as f:
        for line in f.readlines():
            info = line.strip().split('\t')
            person_no = info[person_no_index]
            if person_no in person_no_list:
                print('ERROR: No. %s conflicts with a previous No.. Ignore %s.' % (person_no, info))
                continue
            image_path = os.path.join(args.indir, info[-1])
            if not os.path.exists(image_path):
                print('ERROR: Image path %s not exist. Ignore %s.' % (image_path, info))
                continue
            image_ext = '.' + info[-1].split('.')[-1]
            if image_ext not in args.exts:
                print('ERROR: Image "%s" has unacceptable image extension "%s". '
                      'Ignore %s.' % (info[-1], image_ext, info))
                continue
            person_no_list.append(person_no)
            person_info_list_tmp.append(info)

    person_info_list = []
    for info in person_info_list_tmp:
        image_path = os.path.join(args.indir, info[-1])
        # print(image_path)
        try:
            image = imageio.imread(image_path)
        except (IOError, ValueError, IndexError) as e:
            print('ERROR: Reading %s failed (%s). Ignore %s.' % (info[-1], e, info))
            continue
        if image.ndim < 2:
            print('ERROR: Image dim error in %s. Ignore %s.' % (info[-1], info))
            continue
        if image.ndim == 2:
            image = to_rgb(image)
        image = image[:, :, 0:3]

        # face detection
        with tf.Graph().as_default():
            sess = tf.compat.v1.Session()
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

        minsize = 60
        threshold = [0.6, 0.85, 0.8]
        factor = 0.85
        _minsize = minsize
        _bbox = None
        _landmark = None

        bounding_boxes, points = detect_face.detect_face(image, _minsize, pnet, rnet, onet, threshold, factor)
        # face detection end

        # print('$TEST$ bounding_boxes.shape[0]:', bounding_boxes.shape[0], 'len(points):', len(points))

        # check condition: one face
        faces_num = bounding_boxes.shape[0]
        if faces_num != 1:
            if faces_num == 0:
                print('ERROR: No face detected in image %s. Ignore %s.' % (info[-1], info))
            else:
                print('ERROR: %d face detected in image %s. Ignore %s.' % (faces_num, info[-1], info))
                print('\tNote that only photos of single persons are accepted.')
            continue

        # write 112*112 pictures
        if len(points) > 0:
            _landmark = points.T

        # print(_landmark.shape)
        warped = face_preprocess.preprocess(image,
                                            bbox=bounding_boxes[0],
                                            landmark=_landmark[0].reshape([2, 5]).T,
                                            image_size=args.image_size)
        bgr = warped[..., ::-1]
        # cv2.imshow(str(0), bgr)
        target_file = os.path.join(thumb_dir, info[person_no_index] + '.jpg')
        cv2.imwrite(target_file, bgr)
        person_info_list.append(info)

    # sort the list of valid person information
    def get_person_no(person_info):
        return person_info[person_no_index]
    person_info_list.sort(key=get_person_no)

    # write person information
    with open(person_info_path, 'w', encoding=args.encoding) as f:
        for info in person_info_list:
            info_line = info[person_no_index]
            for item in info[1:-1]:
                info_line += ('\t' + item)
            info_line += ('\t' + info[person_no_index] + '.jpg')
            info_line += '\n'
            f.write(info_line)

    # write binary file of 112*112 pictures
    i = 0
    data_bin = []
    for info in person_info_list:
        image_path = os.path.join(thumb_dir, info[person_no_index] + '.jpg')
        with open(image_path, 'rb') as f:
            _bin = f.read()
            data_bin.append(_bin)
        i += 1
        if i % 1000 == 0:
            print('loading thumb', i)

    with open(data_bin_path, 'wb') as f:
        pickle.dump(data_bin, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
