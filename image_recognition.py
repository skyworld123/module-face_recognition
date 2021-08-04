from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import numpy as np
import imageio
import datetime
from PIL import Image

from config.config import *
from utils.utils import *


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--db',
                        default='db_demo',
                        type=str, help='Database used for face recognition.')

    parser.add_argument('--input',
                        default='input/db_demo_image_input.jpg',
                        type=str, help='Path of input image.')

    parser.add_argument('--output-text',
                        default='output/db_demo_output_text.txt',
                        type=str, help='Path to save the recognition result (text file).')

    parser.add_argument('--output',
                        default='output/db_demo_output.jpg',
                        type=str, help='Path to save the recognition result (image).')

    parser.add_argument('--dup-check',
                        default=1,
                        type=int, help='Set to >=1 to do face duplication check.')

    parser.add_argument('--image-size', type=str, help='Image size (height, width) in pixels.', default='112,112')

    parser.add_argument('--exts', nargs='+', default=['.jpeg', '.jpg', '.png', '.bmp'],
                        help='list of acceptable image extensions.')

    parser.add_argument('--encoding',
                        default='utf-8',
                        help='Encoding of text files.')

    return parser.parse_args(argv)


def get_image_size_tuple(image_size_str):
    size_list = image_size_str.split(',')
    if len(size_list) != 2:
        color_print('ERROR: Format of image size is wrong. Abort.', error_c)
        color_print('\tNote: Correct format: "x,y" (without quotes)', error_c)
        exit(0)
    image_size_tuple = (int(size_list[0]), int(size_list[1]))
    return image_size_tuple


def main(args):
    # initialization
    model_dir = os.path.join(model_root_dir, model_name)
    if not os.path.exists(model_dir):
        color_print('ERROR: Model "%s" not found. Abort.' % model_name, error_c)
        color_print('\tNote: Please put model %s under "model/" before running this programme.' % model_name, error_c)
        exit(0)
    database_dir = os.path.join(db_root_dir, args.db)
    if not os.path.exists(database_dir):
        color_print('ERROR: Database "%s" not exist. Abort.' % args.db, error_c)
        exit(0)
    data_bin_path = os.path.join(database_dir, db_data_bin)
    if not os.path.exists(data_bin_path):
        color_print('ERROR: Binary file "%s" in database not exist. Abort.' % db_data_bin, error_c)
        exit(0)
    person_info_path = os.path.join(database_dir, db_person_info)
    if not os.path.exists(person_info_path):
        color_print('ERROR: Person information file "%s" in database not exist. Abort.' % db_person_info, error_c)
        exit(0)
    if not os.path.exists(args.input):
        color_print('ERROR: Input photo "%s" not exist. Abort.' % args.input, error_c)
        exit(0)
    image_size_tuple = get_image_size_tuple(args.image_size)
    image_ext = '.' + args.input.split('.')[-1]
    if image_ext not in args.exts:
        color_print('ERROR: Input photo "%s" has unacceptable image extension "%s". Abort.' % (args.input, image_ext),
                    error_c)
        exit(0)

    # load model
    color_print('Loading model ...', normal_c)
    time0 = datetime.datetime.now()

    model = load_model(model_name, image_size_tuple)

    time_now = datetime.datetime.now()
    diff = time_now - time0
    color_print('Model loading time: %fs' % diff.total_seconds(), normal_c)
    # load model end

    # load database
    color_print('Loading database %s ...' % args.db, normal_c)
    time0 = datetime.datetime.now()

    db_embed_list, person_info_list = load_database(model, args.db, image_size_tuple, args.encoding)

    time_now = datetime.datetime.now()
    diff = time_now - time0
    color_print('Database loading time: %fs' % diff.total_seconds(), normal_c)
    # load database end

    # parameters of face detection
    color_print('Loading other parameters ...', normal_c)
    det_params, det_params_nets = get_det_params()
    color_print('Loading other parameters finished', normal_c)

    # face recognition
    color_print('Recognizing faces in input image ...', normal_c)
    time0 = datetime.datetime.now()

    input_image = imageio.imread(args.input)
    thumb_image_list, bbox_list = get_detected_faces(input_image, args.image_size, det_params, det_params_nets)
    result_index_list = get_match_list(model, thumb_image_list, db_embed_list, image_size_tuple, args.dup_check)
    result_list = get_person_name(result_index_list, person_info_list)

    write_result_text(result_list, bbox_list, args.output_text, args.encoding)

    pil_image = Image.open(args.input)
    result_image = get_result_image(pil_image, result_list, result_index_list, bbox_list)
    result_image.save(args.output)

    time_now = datetime.datetime.now()
    diff = time_now - time0
    color_print('Face recognition time: %fs' % diff.total_seconds(), normal_c)
    # face recognition end


if __name__ == '__main__':
    os.system('')  # for prettifying console output

    main(parse_arguments(sys.argv[1:]))
