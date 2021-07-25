"""
configurations
"""

# database
db_root_dir = 'database'
db_thumb_dir = 'thumb'
db_person_info = 'person_info.txt'
db_data_bin = 'data_bin.bin'
output_thumb_base_dir = 'others/face_thumb'
person_no_index = 0
person_name_index = 1

# model
model_root_dir = 'models'
model_name = 'model-r100-ii'
batch_size = 16
num_folds = 10

# threshold for matching embeddings
match_threshold = 1.50

# others
unknown_person_str = '[Unknown person]'
shown_name_magnif = 0.2  # magnification of shown name,
# recommend to choose a value between 0.1 (small) and 0.3 (large)
shown_name_min_size = 10
# shown_name_font = 'simsun.ttc'
shown_name_font = 'C:\\Windows\\Fonts\\msyh.ttc'  # "微软雅黑"
shown_name_max_len = 15
