# module-face_recognition

采用InsightFace人脸识别模型和MTCNN人脸检测算法实现的人脸识别模块。

## 使用说明

**make_database.py**

创建包含已知人脸的数据库。所有人脸照片放在一个文件夹下。对于每个人需要给出其编号（关键字）、姓名和照片名称信息，输入到文本文件中。

输入参数：

* --db: 数据库名称。
* --create: 是否创建新数据库，若为False则无法对不存在的数据库进行操作。
* --indir: 人脸照片所在文件夹，每张照片只能包含一张要输入数据库的已知人脸。
* --input-text: 所有人的编号、姓名和照片名称（仅输入在indir下的名称）。
* --image-size: 使用MTCNN对输入的人脸照片进行裁剪，得到的人脸缩略图的尺寸，默认缩略图尺寸为112 * 112。
* --exts: 人脸照片可选拓展名，默认包含.jpeg, .jpg, .png, .bmp格式。
* --encoding: 打开文本文件所使用的的编码，默认为utf-8。

**face_recognition.py**

使用已有数据库对给定图像进行人脸识别。

输入参数：

* --db: 已有数据库名称。
* --input: 要进行人脸识别的图片路径。
* --output-text: 输出的包含人脸识别结果信息（识别出的人脸序号、人脸位置）的文本文件的路径。
* --output-image: 输出的人脸识别结果图片的路径。
* --image-size: 使用MTCNN对输入的人脸照片进行裁剪，得到的人脸缩略图的尺寸，默认缩略图尺寸为112 * 112。
* --exts: 人脸照片可选拓展名，默认包含.jpeg, .jpg, .png, .bmp格式。
* --encoding: 打开文本文件所使用的的编码，默认为utf-8。

**注意：** 运行代码前，请事先从InsightFace的[Model-Zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo)下载所需的人脸识别模型，并放置在model/目录下。（如果你愿意，也可以根据InsightFace中训练人脸识别模型的步骤自行训练模型并使用之。）本项目默认使用Model-Zoo中的LResNet100E-IR,ArcFace@ms1m-refine-v2（model-r100-ii），若选用其他模型，请在config/config.py中修改model_name配置。

