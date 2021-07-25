from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import cv2
import imageio
import pickle
import mxnet as mx
from mxnet import ndarray as nd
import datetime
import sklearn
from sklearn import preprocessing
from PIL import Image, ImageDraw, ImageFont

from config.config import *
from face_detection import detect_face, face_image, face_preprocess


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--db',
                        default='db1',
                        type=str, help='Database used for face recognition.')

    parser.add_argument('--input',
                        default='input/db1_rec_input.jpg',
                        type=str, help='Input photo.')

    parser.add_argument('--output-text',
                        default='output/db1_output_text.txt',
                        type=str, help='Output result in text file.')

    parser.add_argument('--output-image',
                        default='output/db1_output_image.jpg',
                        type=str, help='Output result in image file.')

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


def load_model(model_name, image_size):
    ctx = mx.gpu(0)
    model_dir = os.path.join(model_root_dir, model_name)
    model_dir_extended = model_dir + '/'

    epochs = []
    for file in os.listdir(model_dir):
        if not file.endswith('.params'):
            continue
        _epoch = int(file.split('.')[0].split('-')[1])
        epochs.append(_epoch)
    if len(epochs) == 0:
        print('ERROR: No model found in "%s/". Abort.' % model_root_dir)
        exit(0)
    epochs = sorted(epochs, reverse=True)
    epoch = epochs[0]

    sym, arg_params, aux_params = mx.model.load_checkpoint(model_dir_extended, epoch)
    all_layers = sym.get_internals()
    sym = all_layers['fc1_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=[])
    model.bind(data_shapes=[('data', (batch_size, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)

    return model


def load_bin(bin_list, image_size):
    # get the combination of original and reversed image
    # may need to enlarge the data to adjust to the model
    data_list = []
    data_size = max(len(bin_list), batch_size)  # possible enlargement

    for i in range(len([0, 1])):
        data = nd.empty((data_size, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for i in range(len(bin_list)):
        _bin = bin_list[i]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                # reversed image
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][i][:] = img
        if i > 0 and i % 1000 == 0:
            print('loading bin', i)
    # print('$TEST$ (in load_bin) data_list[0].shape:', data_list[0].shape)
    return data_list


def get_embeddings(model, image_comb_list, image_num):
    embeddings_list = []

    image_input_num = image_comb_list[0].shape[0]
    assert image_input_num >= batch_size
    assert image_num >= 0
    image_num = min(image_num, image_input_num)

    _label = nd.ones((batch_size,))

    for i in range(len(image_comb_list)):  # i = 0, 1
        data = image_comb_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = nd.slice_axis(data, axis=0, begin=bb - batch_size, end=bb)  # key step 1
            # print('$TEST$ in get_embeddings', _data.shape, _label.shape)
            db = mx.io.DataBatch(data=(_data,), label=(_label,))  # key step 2
            model.forward(db, is_train=False)  # vital step, key step 3
            net_out = model.get_outputs()  # vital step, key step 4
            _embeddings = net_out[0].asnumpy()  # vital step, key step 5
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]  # key step 6
            ba = bb
        embeddings_list.append(embeddings)  # key step 7

    embeddings = embeddings_list[0] + embeddings_list[1]  # key step 8
    embeddings = preprocessing.normalize(embeddings)  # vital step, key step 9

    return embeddings[:image_num]


def best_match(db_embed_list, input_embed):
    input_embed_cp_list = [input_embed] * len(db_embed_list)
    diff = np.subtract(db_embed_list, input_embed_cp_list)
    dist = np.sum(np.square(diff), 1)
    # print('$TEST$ dist:', dist)
    best_index = np.argmin(dist)
    # print('$TEST$ in best_match: best index:', best_index, dist[best_index])
    if dist[best_index] < match_threshold:
        return best_index
    else:
        return None


def match_embeddings(db_embed_list, input_embed_list):
    result = []
    for input_embed in input_embed_list:
        best_index = best_match(db_embed_list, input_embed)
        result.append(best_index)
    return result


def get_image_size_tuple(image_size_str):
    size_list = image_size_str.split(',')
    if len(size_list) != 2:
        print('ERROR: Format of image size is wrong. Abort.'
              '\tNote: Correct format: "x,y" (without quotes)')
        exit(0)
    image_size_tuple = (int(size_list[0]), int(size_list[1]))
    return image_size_tuple


def main(args):
    # initialization
    database_dir = os.path.join(db_root_dir, args.db)
    if not os.path.exists(database_dir):
        print('ERROR: Database "%s" not exist. Abort.' % args.db)
        exit(0)
    person_info_path = os.path.join(database_dir, db_person_info)
    data_bin_path = os.path.join(database_dir, db_data_bin)
    if not os.path.exists(args.input):
        print('ERROR: Input photo "%s" not exist. Abort.' % args.input)
        exit(0)
    image_size_tuple = get_image_size_tuple(args.image_size)
    image_ext = '.' + args.input.split('.')[-1]
    if image_ext not in args.exts:
        print('ERROR: Input photo "%s" has unacceptable image extension "%s". Abort.' % (args.input, image_ext))
        exit(0)

    # load model
    print('Loading model ...')
    time0 = datetime.datetime.now()
    model = load_model(model_name, image_size_tuple)
    time_now = datetime.datetime.now()
    diff = time_now - time0
    print('Model loading time:', diff.total_seconds())

    # load database
    print('Loading database %s ...' % args.db)
    if not os.path.exists(data_bin_path):
        print('ERROR: Binary file "%s" in database not exist. Abort.' % db_data_bin)
        exit(0)

    time0 = datetime.datetime.now()

    # load binary file and calculate embeddings
    try:
        with open(data_bin_path, 'rb') as f:
            bin_list = pickle.load(f)  # py2
    except UnicodeDecodeError:
        with open(data_bin_path, 'rb') as f:
            bin_list = pickle.load(f, encoding='bytes')  # py3
    db_image_num = len(bin_list)
    image_comb_list = load_bin(bin_list, image_size_tuple)
    db_embed_list = get_embeddings(model, image_comb_list, db_image_num)

    # load person information
    person_info_list = []
    with open(person_info_path, 'r', encoding=args.encoding) as f:
        for line in f.readlines():
            info = line.strip().split('\t')
            person_info_list.append(info)

    time_now = datetime.datetime.now()
    diff = time_now - time0
    print('Database loading time:', diff.total_seconds())
    # load database end

    # face recognition
    time0 = datetime.datetime.now()
    try:
        image = imageio.imread(args.input)
    except (IOError, ValueError, IndexError) as e:
        print('ERROR: Reading input photo %s failed (%s). Abort.' % (args.input, e))
        exit(0)
    else:
        if image.ndim < 2:
            print('ERROR: Image dim error in input photo %s. Abort.' % args.input)
            exit(0)
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

        if len(points) > 0:
            _landmark = points.T

        # write 112*112 pictures
        output_thumb_dir = os.path.join(output_thumb_base_dir, args.db)
        if not os.path.exists(output_thumb_dir):
            os.makedirs(output_thumb_dir)
        faces_num = bounding_boxes.shape[0]
        thumb_list = []
        for num in range(faces_num):
            warped = face_preprocess.preprocess(image,
                                                bbox=bounding_boxes[num],
                                                landmark=_landmark[num].reshape([2, 5]).T,
                                                image_size=args.image_size)
            bgr = warped[..., ::-1]
            input_image_name = os.path.basename(args.input).split('.')[0]
            target_file = os.path.join(output_thumb_dir, input_image_name + '-%04d.jpg' % (num + 1))
            thumb_list.append(target_file)
            cv2.imwrite(target_file, bgr)

        # get binary data of faces in input photo
        input_bin_list = []
        for thumb_path in thumb_list:
            with open(thumb_path, 'rb') as f:
                _bin = f.read()
                input_bin_list.append(_bin)
        input_image_num = len(input_bin_list)
        input_image_comb_list = load_bin(input_bin_list, image_size_tuple)
        input_embed_list = get_embeddings(model, input_image_comb_list, input_image_num)

        # match the best result
        result_index_list = match_embeddings(db_embed_list, input_embed_list)

        # get result
        result_list = []
        for result_index in result_index_list:
            if result_index is None:
                result_list.append(unknown_person_str)
            else:
                result_list.append(person_info_list[result_index][person_name_index])

        # write result (text)
        with open(args.output_text, 'w', encoding=args.encoding) as f:
            summary = 'Total result: %d faces\n\n' % len(result_list)
            f.write(summary)
            for i in range(len(bounding_boxes)):
                line = str(i + 1) + '\t' + str(bounding_boxes[i][0:4]) + '\t' + str(result_list[i])
                line += '\n'
                f.write(line)

        # write result (image)
        result_image = Image.open(args.input)
        draw = ImageDraw.Draw(result_image)
        for i in range(len(bounding_boxes)):
            bbox_pos = (int(bounding_boxes[i][0]), int(bounding_boxes[i][1]),
                        int(bounding_boxes[i][2]), int(bounding_boxes[i][3]))
            known_color = (0, 0, 255)  # blue, RGB
            unknown_color = (0, 204, 204)  # orange, RGB
            name_color = (255, 255, 255)  # white, RGB
            bbox_width = bbox_pos[2] - bbox_pos[0]
            known_rect_width = max(int(bbox_width * 0.04), 1)
            unknown_rect_width = max(int(bbox_width * 0.02), 1)

            if result_index_list[i] is None:
                rect_color = unknown_color
                rect_width = unknown_rect_width
            else:
                rect_color = known_color
                rect_width = known_rect_width
            draw.rectangle(bbox_pos, None, rect_color, rect_width)

            if not result_index_list[i] is None:
                font_size = max(int(bbox_width * shown_name_magnif), shown_name_min_size)
                margin = max(int(font_size * 0.1), 1)
                font = ImageFont.truetype(shown_name_font, font_size)
                output_name = result_list[i][:shown_name_max_len]
                if len(result_list[i]) > shown_name_max_len:
                    output_name += '...'
                text_size = font.getsize(output_name)
                bgr_pos_ur = (min(bbox_pos[0] + text_size[0] + margin * 2, result_image.size[0]),
                              max(bbox_pos[1] - font_size - margin * 2, 0))
                bgr_pos = (bgr_pos_ur[0] - text_size[0] - margin * 2, bgr_pos_ur[1],
                           bgr_pos_ur[0], bgr_pos_ur[1] + text_size[1] + margin * 2)
                draw.rectangle(bgr_pos, known_color, known_color)
                text_pos_ul = (bgr_pos[0] + margin, bgr_pos[1] + margin)
                draw.text(text_pos_ul, output_name, name_color, font)

        result_image.save(args.output_image)

    time_now = datetime.datetime.now()
    diff = time_now - time0
    print('Face recognition time:', diff.total_seconds())


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
