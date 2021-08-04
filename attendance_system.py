from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import numpy as np
import datetime
import cv2
from PIL import Image
import copy

from config.config import *
from utils.utils import *


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--db',
                        default='db_demo',
                        type=str, help='Database used for face recognition.')

    parser.add_argument('--camera',
                        default=1,
                        type=int, help='Set to >=1 to read video from the front camera on your computer, '
                                       'otherwise read video from local files.')

    parser.add_argument('--input',
                        default='input/input_video.mp4',
                        type=str, help='Path of input video (if "--camera" set to 0).')

    parser.add_argument('--ckp-text',
                        default='input/checked_persons.txt',
                        type=str, help='Path of input text that contains checked persons.')

    parser.add_argument('--check-res',
                        default='output/check_result.txt',
                        type=str, help='Path of output text that contains the result of attendance check.')

    parser.add_argument('--save',
                        default=0,
                        type=int, help='Set to >=1 to save the recognition result (video).')

    parser.add_argument('--output',
                        default='output/video_rec_result.avi',
                        type=str, help='Path to save the recognition result (video) (if "--save" set to >=1).')

    parser.add_argument('--dup-check',
                        default=1,
                        type=int, help='Set to >=1 to do face duplication check.')

    parser.add_argument('--image-size', type=str, help='Image size (height, width) in pixels.', default='112,112')

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


def get_checked_persons(text_path, person_info_list, encoding='utf-8'):
    ckp_idx_list = []

    known_no_list = [info[person_no_index] for info in person_info_list]
    with open(text_path, 'r', encoding=encoding) as f:
        for line in f.readlines():
            line_strip = line.strip()
            if line_strip == '':  # blank lines in input text
                continue
            elif line_strip.startswith('#'):  # comments in input text
                continue
            if line_strip in known_no_list:
                ckp_no = known_no_list.index(line_strip)
                ckp_idx_list.append(ckp_no)

    ckp_idx_list.sort()
    return ckp_idx_list


def get_db_to_ckp(ckp_idx_list, db_person_num):
    db_to_ckp = [-1] * db_person_num
    for ckp_i, db_i in enumerate(ckp_idx_list):
        db_to_ckp[db_i] = ckp_i
    return db_to_ckp


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
    if args.camera <= 0 and not os.path.exists(args.input):
        color_print('ERROR: Input video "%s" not exist. Abort.' % args.input, error_c)
        exit(0)
    if not os.path.exists(args.ckp_text):
        color_print('ERROR: Input checked persons text "%s" not exist. Abort.' % args.ckp_text, error_c)
        exit(0)
    image_size_tuple = get_image_size_tuple(args.image_size)

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

    # get checked persons
    ckp_idx_list = get_checked_persons(args.ckp_text, person_info_list, args.encoding)

    # get index list from database index to ckp index
    db_to_ckp = get_db_to_ckp(ckp_idx_list, len(person_info_list))

    # necessary information for attendance system
    default_dup_check = args.dup_check
    dup_check = default_dup_check

    ckp_status_list = [un_regis] * len(ckp_idx_list)
    ckp_time_list = [0] * len(ckp_idx_list)
    check_record = []
    sys_status = sys_default

    this_record_begin_t = datetime.datetime.now()
    this_record_end_t = datetime.datetime.now()
    this_record_person_idx_list = []

    registering_person_list = []
    registering_person_prev_list = []

    color_print('Loading other parameters finished', normal_c)

    # face recognition
    color_print('Opening attendance system ...', normal_c)
    if args.camera >= 1:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.input)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS) / frame_ratio)
    if args.save >= 1:
        out = cv2.VideoWriter(args.output, fourcc, fps, size)

    show_size = size
    if show_size[0] > size_max_h:
        show_size = (size_max_h, int(show_size[1] * size_max_h / show_size[0]))
    if show_size[1] > size_max_v:
        show_size = (int(show_size[0] * size_max_v / show_size[1]), size_max_v)

    regis_frame_cnt = fps * regis_sec

    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cnt % frame_ratio == 0:
                # face recognition for input image
                thumb_image_list, bbox_list = get_detected_faces(frame, args.image_size, det_params, det_params_nets)
                result_index_list = get_match_list(model, thumb_image_list, db_embed_list, image_size_tuple,
                                                   dup_check)
                result_list = get_person_name(result_index_list, person_info_list)

                # intermediate process of attendance system
                result_status_list = [free] * len(result_index_list)
                if sys_status != sys_default:
                    for i, idx in enumerate(result_index_list):
                        ckp_idx = db_to_ckp[idx]
                        if ckp_idx != -1:
                            # change status
                            if sys_status == sys_check_on:
                                if ckp_time_list[ckp_idx] < regis_frame_cnt:
                                    ckp_time_list[ckp_idx] += 1
                                    ckp_status_list[ckp_idx] = registering
                                    if ckp_time_list[ckp_idx] == regis_frame_cnt:
                                        this_record_person_idx_list.append(idx)
                                        ckp_status_list[ckp_idx] = regis
                                    else:
                                        registering_person_list.append(idx)

                            # get result_status_list
                            result_status_list[i] = ckp_status_list[ckp_idx]

                    # change status of persons who disappear early
                    if sys_status == sys_check_on:
                        for p in registering_person_prev_list:
                            if p not in registering_person_list:
                                ckp_idx = db_to_ckp[p]
                                if ckp_status_list[ckp_idx] == registering:
                                    ckp_status_list[ckp_idx] = un_regis
                                    ckp_time_list[ckp_idx] = 0
                        registering_person_prev_list = copy.copy(registering_person_list)
                        registering_person_list.clear()

                # draw output image
                pil_image = Image.fromarray(frame)
                if sys_status == sys_default:
                    result_image = get_result_image(pil_image, result_list, result_index_list, bbox_list, True)
                else:
                    result_image = get_result_image_sys(pil_image, result_list, result_index_list, result_status_list,
                                                        bbox_list, True)

                # window resizing
                result_image_np = np.array(result_image)
                result_show = cv2.resize(result_image_np, show_size)

                cv2.imshow('result-show', result_show)

                if args.save >= 1:
                    out.write(result_image_np)
        else:
            break

        cnt += 1
        key = cv2.waitKey(1)
        key_f = key & 0xff
        if key_f == ord('q'):  # press 'q' to quit manually
            break

        # control keys
        if key_f == ord('a'):  # open or close attendance check
            if sys_status == sys_default:
                print('Enter status sys_check_ready')
                sys_status = sys_check_ready
                dup_check = True
            elif sys_status == sys_check_ready:
                print('Quit status sys_check_ready, enter status sys_default')
                sys_status = sys_default
                dup_check = default_dup_check

        elif key_f == ord('z'):  # begin registration
            if sys_status == sys_check_ready:
                print('Enter status sys_check_on')
                sys_status = sys_check_on
                this_record_begin_t = datetime.datetime.now()

        elif key_f == ord('x'):  # end registration
            if sys_status == sys_check_on:
                print('Quit status sys_check_on, enter status sys_check_ready')
                sys_status = sys_check_ready
                this_record_end_t = datetime.datetime.now()
                t_r_name_list = get_person_name(this_record_person_idx_list, person_info_list)
                check_record.append([this_record_begin_t, this_record_end_t, len(t_r_name_list), t_r_name_list])
                this_record_person_idx_list.clear()
                for i, s in enumerate(ckp_status_list):
                    if s == registering:  # clear the appearance time of unregistered persons
                        ckp_status_list[i] = un_regis
                        ckp_time_list[i] = 0

        elif key_f == ord('w'):  # write check result
            if sys_status == sys_check_ready:
                print('Write check result to %s' % args.check_res)
                with open(args.check_res, 'w') as f:
                    f.write('Total check periods: %d\n\n' % len(check_record))
                    for record in check_record:
                        f.write(str(record) + '\n')
                    f.write('\n')
                    present_idx_list = []
                    absent_idx_list = []
                    for i, s in enumerate(ckp_status_list):
                        if s == regis:
                            present_idx_list.append(ckp_idx_list[i])
                        else:
                            absent_idx_list.append(ckp_idx_list[i])
                    present_name_list = get_person_name(present_idx_list, person_info_list)
                    absent_name_list = get_person_name(absent_idx_list, person_info_list)
                    f.write('Final check result:\n')
                    f.write('Present persons: ' + str(len(present_name_list)) + ', ' + str(present_name_list) + '\n')
                    f.write('Absent persons: ' + str(len(absent_name_list)) + ', ' + str(absent_name_list) + '\n')

        elif key_f == ord('c'):  # clear check result
            if sys_status == sys_check_ready:
                print('Clear check result')
                ckp_status_list = [un_regis] * len(ckp_idx_list)
                ckp_time_list = [0] * len(ckp_idx_list)
                check_record.clear()

        elif key_f == ord('v'):  # change duplication check setting
            if sys_status == sys_default:
                if default_dup_check >= 1:
                    default_dup_check = 0
                else:
                    default_dup_check = 1
                dup_check = default_dup_check
                if dup_check >= 1:
                    print('Duplication check on')
                else:
                    print('Duplication check off')
            else:
                color_print('Change duplication check setting failed', warning_c)
                color_print('\tNote: When attendance check is open, duplication check must be on (>=1)', warning_c)

    cap.release()
    if args.save >= 1:
        out.release()
    cv2.destroyAllWindows()
    color_print('Close attendance system', normal_c)
    # face recognition end


if __name__ == '__main__':
    os.system('')  # for prettifying console output

    main(parse_arguments(sys.argv[1:]))
