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

# public interfaces
__all__ = ['load_model', 'load_database', 'get_embeddings', 'get_det_params', 'get_detected_faces',
           'get_match_list', 'get_person_name', 'write_result_text', 'get_result_image', 'get_result_image_sys']


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def image2comb(image_list, image_size):
    # get the combination of original and reversed image from raw images (in mxnet ndarray)
    # may need to enlarge the data to adjust to the model
    data_list = []
    data_size = max(len(image_list), batch_size)  # possible enlargement

    for i in range(len([0, 1])):
        data = nd.empty((data_size, 3, image_size[0], image_size[1]))
        data_list.append(data)

    for i in range(len(image_list)):
        img = image_list[i]
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                # reversed image
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][i][:] = img
        if i > 0 and (i + 1) % 1000 == 0:
            print('processing face thumbnails (%d/%d)' % (i + 1, len(image_list)))

    return data_list


def match_embeddings_with_check(db_embed_list, input_embed_list):
    result = [-1] * len(input_embed_list)

    match_list = []
    input_ack = []
    db_ack = []

    for x, input_embed in enumerate(input_embed_list):
        input_embed_copy_list = [input_embed] * len(db_embed_list)
        diff = np.subtract(db_embed_list, input_embed_copy_list)
        dist = np.sum(np.square(diff), 1)
        for n, d in enumerate(dist):
            if d < match_threshold:
                match_list.append([(x, n), d])

    def get_dist(match_list_item):
        return match_list_item[1]
    match_list.sort(key=get_dist)

    for i, item in enumerate(match_list):
        x = item[0][0]
        n = item[0][1]
        if (x not in input_ack) and (n not in db_ack):
            result[x] = n
            input_ack.append(x)
            db_ack.append(n)

    return result


def match_embeddings(db_embed_list, input_embed_list):
    result = [-1] * len(input_embed_list)

    for x, input_embed in enumerate(input_embed_list):
        input_embed_copy_list = [input_embed] * len(db_embed_list)
        diff = np.subtract(db_embed_list, input_embed_copy_list)
        dist = np.sum(np.square(diff), 1)
        min_dist = dist[0]
        min_idx = 0
        for i in range(1, len(dist)):
            d = dist[i]
            if d < min_dist and d < match_threshold:
                min_dist = d
                min_idx = i
        if min_dist < match_threshold:
            result[x] = min_idx

    return result


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
        color_print('ERROR: Necessary .params file not found in "%s". Abort.' % model_dir, error_c)
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


def load_database(model, database, image_size_tuple, encoding='utf-8'):
    database_dir = os.path.join(db_root_dir, database)
    data_bin_path = os.path.join(database_dir, db_data_bin)
    person_info_path = os.path.join(database_dir, db_person_info)

    try:
        with open(data_bin_path, 'rb') as f:
            bin_list = pickle.load(f)  # py2
    except UnicodeDecodeError:
        with open(data_bin_path, 'rb') as f:
            bin_list = pickle.load(f, encoding='bytes')  # py3
    image_list = []
    for _bin in bin_list:
        img = mx.image.imdecode(_bin)
        image_list.append(img)
    db_image_num = len(image_list)
    image_comb_list = image2comb(image_list, image_size_tuple)
    db_embed_list = get_embeddings(model, image_comb_list, db_image_num)

    # load person information
    person_info_list = []
    with open(person_info_path, 'r', encoding=encoding) as f:
        for line in f.readlines():
            info = line.strip().split('\t')
            person_info_list.append(info)

    return db_embed_list, person_info_list


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


def get_det_params():
    minsize = 60
    threshold = [0.6, 0.85, 0.8]
    factor = 0.85
    _minsize = minsize
    _bbox = None
    _landmark = None
    det_params = (_minsize, threshold, factor)

    with tf.Graph().as_default():
        sess = tf.compat.v1.Session()
        with sess.as_default():
            det_params_nets = detect_face.create_mtcnn(sess, None)

    return det_params, det_params_nets


def get_detected_faces(image, image_size, params, params_nets):
    _minsize, threshold, factor = params
    pnet, rnet, onet = params_nets

    if image.ndim < 2:
        color_print('ERROR: Image dim error in input image. Abort.', error_c)
        exit(0)
    if image.ndim == 2:
        image = to_rgb(image)
    image = image[:, :, 0:3]

    # face detection
    bbox_list, points = detect_face.detect_face(image, _minsize, pnet, rnet, onet, threshold, factor)
    # face detection end

    if len(points) > 0:
        _landmark = points.T

    # write 112*112 pictures
    faces_num = bbox_list.shape[0]
    thumb_image_list = []
    for num in range(faces_num):
        warped = face_preprocess.preprocess(image,
                                            bbox=bbox_list[num],
                                            landmark=_landmark[num].reshape([2, 5]).T,
                                            image_size=image_size)
        bgr = warped[..., ::-1]
        bgr_nd = mx.nd.array(bgr)
        thumb_image_list.append(bgr_nd)

    return thumb_image_list, bbox_list


def get_match_list(model, thumb_image_list, db_embed_list, image_size_tuple, dup_check):
    # get embeddings of faces in input photo
    input_image_num = len(thumb_image_list)
    input_image_comb_list = image2comb(thumb_image_list, image_size_tuple)
    input_embed_list = get_embeddings(model, input_image_comb_list, input_image_num)

    # match the best result
    if dup_check >= 1:
        result_index_list = match_embeddings_with_check(db_embed_list, input_embed_list)
    else:
        result_index_list = match_embeddings(db_embed_list, input_embed_list)

    return result_index_list


def get_person_name(index_list, person_info_list):
    name_list = []
    for index in index_list:
        if index == -1:
            name_list.append(unknown_person_str)
        else:
            name_list.append(person_info_list[index][person_name_index])
    return name_list


def write_result_text(result_list, bbox_list, output_text_path, encoding='utf-8'):
    with open(output_text_path, 'w', encoding=encoding) as f:
        summary = 'Total result: %d faces\n\n' % len(result_list)
        f.write(summary)
        for i in range(len(bbox_list)):
            line = str(i + 1) + '\t' + str(bbox_list[i][0:4]) + '\t' + str(result_list[i])
            line += '\n'
            f.write(line)


def get_result_image(original_image, result_list, result_index_list, bbox_list, convert_rgb=False):
    # note: input "original_image" must be a PIL image
    result_image = original_image
    draw = ImageDraw.Draw(result_image)
    for i in range(len(bbox_list)):
        bbox_pos = (int(bbox_list[i][0]), int(bbox_list[i][1]),
                    int(bbox_list[i][2]), int(bbox_list[i][3]))
        bbox_width = bbox_pos[2] - bbox_pos[0]
        known_rect_width = max(int(bbox_width * 0.04), 1)
        unknown_rect_width = max(int(bbox_width * 0.02), 1)

        if result_index_list[i] == -1:
            rect_color = unknown_color
            rect_width = unknown_rect_width
        else:
            rect_color = known_color
            rect_width = known_rect_width
        name_bg_color = rect_color
        name_color = p_name_color

        def get_convert(color):
            return color[2], color[1], color[0]
        if convert_rgb:
            rect_color = get_convert(rect_color)
            name_bg_color = get_convert(name_bg_color)
            name_color = get_convert(p_name_color)

        # draw the bounding box
        draw.rectangle(bbox_pos, None, rect_color, rect_width)

        if result_index_list[i] != -1:
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
            # draw the background of a person's name
            draw.rectangle(bgr_pos, name_bg_color, name_bg_color)
            text_pos_ul = (bgr_pos[0] + margin, bgr_pos[1] + margin)
            # draw a person's name
            draw.text(text_pos_ul, output_name, name_color, font)

    return result_image


# get_result_image for attendance system
def get_result_image_sys(original_image, result_list, result_index_list, result_status_list, bbox_list,
                         convert_rgb=False):
    # note: input "original_image" must be a PIL image
    result_image = original_image
    draw = ImageDraw.Draw(result_image)
    for i in range(len(bbox_list)):
        bbox_pos = (int(bbox_list[i][0]), int(bbox_list[i][1]),
                    int(bbox_list[i][2]), int(bbox_list[i][3]))
        bbox_width = bbox_pos[2] - bbox_pos[0]
        known_rect_width = max(int(bbox_width * 0.04), 1)
        unknown_rect_width = max(int(bbox_width * 0.02), 1)

        if result_index_list[i] == -1:
            rect_color = unknown_color
            rect_width = unknown_rect_width
        else:
            status = result_status_list[i]
            if status == un_regis:
                rect_color = un_regis_color
            elif status == registering:
                rect_color = registering_color
            elif status == regis:
                rect_color = regis_color
            else:
                rect_color = known_color
            rect_width = known_rect_width
        name_bg_color = rect_color
        name_color = p_name_color

        def get_convert(color):
            return color[2], color[1], color[0]
        if convert_rgb:
            rect_color = get_convert(rect_color)
            name_bg_color = get_convert(name_bg_color)
            name_color = get_convert(p_name_color)

        # draw the bounding box
        draw.rectangle(bbox_pos, None, rect_color, rect_width)

        if result_index_list[i] != -1:
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
            # draw the background of a person's name
            draw.rectangle(bgr_pos, name_bg_color, name_bg_color)
            text_pos_ul = (bgr_pos[0] + margin, bgr_pos[1] + margin)
            # draw a person's name
            draw.text(text_pos_ul, output_name, name_color, font)

    return result_image
