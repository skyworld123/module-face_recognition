"""
configurations
"""

# database
db_root_dir = 'database'
db_thumb_dir = 'thumb'
db_person_info = 'person_info.txt'
db_data_bin = 'data_bin.bin'
person_no_index = 0
person_name_index = 1

# model
model_root_dir = 'models'
model_name = 'model-r100-ii'
batch_size = 16
num_folds = 10

# threshold for matching embeddings
match_threshold = 1.50

# video recognition
size_max_h = 1280
size_max_v = 720
frame_ratio = 5

# attendance system
# status of the system
sys_default = 0
sys_check_ready = 1
sys_check_on = 2
# status of persons
free = 0
un_regis = 1
registering = 2
regis = 3
# registration time in seconds
regis_sec = 2

# showing result
unknown_person_str = '[Unknown person]'
shown_name_magnif = 0.2  # magnification of shown name,
# recommend to choose a value between 0.1 (small) and 0.3 (large)
shown_name_min_size = 10
# shown_name_font = 'simsun.ttc'
shown_name_font = 'C:\\Windows\\Fonts\\msyh.ttc'  # "微软雅黑"
shown_name_max_len = 15
known_color = (0, 0, 255)  # blue, RGB
unknown_color = (0, 204, 204)  # orange, RGB
p_name_color = (255, 255, 255)  # white, RGB
un_regis_color = (255, 165, 0)  # orange, RGB
registering_color = (255, 215, 0)  # gold, RGB
regis_color = (50, 205, 50)  # lime green, RGB

# console output
def color_print(text, color):
    print(color + text + '\033[0m')

error_c = '\033[31m'
normal_c = '\033[32m'
warning_c = '\033[33m'
