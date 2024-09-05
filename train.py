if __name__ == '__main__':

    from ultralytics import YOLO
    import time
    from pylab import mpl
    import matplotlib



# yolov8m模型训练：训练模型的数据为'Profile.yaml'，轮数为100，图片大小为640，设备为本地的GPU显卡，关闭多线程的加载，图像加载的批次大小为4，开启图片缓存

#model = YOLO("Profile.yaml")  # build a new model from scratch
#YOLO("yolov8m-pose.pt")  # load a pretrained model (recommended for training)
#model = YOLO("Profile02.yaml")  # build a new model from YAML
#model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)
#model = YOLO("Profile02.yaml").load("yolov8m.pt")  # build from YAML and transfer weights

    model = YOLO("yolov8m.pt")  # build a new model from scratch
    model.load("yolov8m.pt")  # load a pretrained model (recommended for training)
#model.train(data="Profile.yaml", epochs=100, imgsz=640, device=0, resume=True, lr0=0.01, batch=16)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU编号
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    # 设置显示中文字体
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    # 设置正常显示符号
    mpl.rcParams["axes.unicode_minus"] = False
# Train the model with 1 GPUs
    results = model.train(data="Profile.yaml", epochs=100, imgsz=640, device=0, workers=2, batch=4, cache=True) # 开始训练

    time.sleep(5) # 睡眠5s，主要是用于服务器多次训练的过程中使用