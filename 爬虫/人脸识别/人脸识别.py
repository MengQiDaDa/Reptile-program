from time import sleep

import cv2
import numpy as np
from PIL import Image
from imutils import face_utils,resize

try:
    from dlib import get_frontal_face_detector,shape_predictor
except ImportError:
    raise

class DynamicStreamMaskService(object):
    """
    创建面具加载服务类DynamicStreamMaskService及其对应的初始化属性：
    动态黏贴面具服务
    """

    def __init__(self,saved=False):
        self.saved = saved  #是否保存图片
        self.listener = True #启动参数
        self.video_capture = cv2.VideoCapture(0) #调用本地摄像头
        self.doing = False #是否进行面部面具
        self.speed = 0.1 #面具移动速度
        self.detector = get_frontal_face_detector()  #面部识别器
        self.predictor = shape_predictor("shape_predictor_68_dace_landmarks.dat")  #面部分析器
        self.fps = 4 #面具存在的基础时间
        self.animation_time = 0 #动画周期初始值
        self.duration = self.fps * 4  #动画周期最大值
        self.fixed_time = 4 #画图之后停留时间
        self.max_width = 500 #图像大小
        self.deal, self.text, self.cigarette = None,None,None #面具对象

    def read_data(self):
        """
        按照上面的介绍，我们先实现读取视频流转换图片的函数:
        从摄像头获取视频流，并转换为一帧一帧的图像
        :return: 返回一帧一帧的图像信息
        """
        _,data = self.video_capture.read()
        return data

    def get_glasses_info(self, face_shape, face_width):
        """
        接下来我们实现人脸定位函数， 及眼镜和烟卷的定位：
        获取当前面部眼睛的信息
        :param face_shape:
        :param face_width:
        :return:
        """

        left_eye = face_shape[36:42]
        right_eye = face_shape[42:48]

        left_eye_center = left_eye.mean(axis = 0).astype("int")
        right_eye_center = right_eye.mean(axis = 0).astype("int")

        y = left_eye_center[1] - right_eye_center[1]
        x = left_eye_center[0] - right_eye_center[0]
        eye_angle = np.rad2deg(np.arctan2(y,x))

        deal = self.deal.resize(
            (face_shape, int(face_width * self.deal.size[1] / self.deal.size[0])),
            resample=lmage.LANCZOS)

        deal = deal.rotate(eye_angle, expand=True)
        deal = deal.transpose(lmage.FLIP_TOP_BOTTOM)

        left_eye_x = left_eye[0,0] - face_width // 4
        left_eye_y = left_eye[0,1] - face_width // 6

        return {"image" : deal, "pos":(left_eye_x,left_eye_y)}

    def get_cigarette_info(self, face_shape, face_width):
        """
        获取当前面部烟卷信息
        :param face_shape:
        :param face_width:
        :return:
        """
        mouth = face_shape[49:68]
        mouth_center = mouth.mean(axis=0).astype("int")
        cigarette = self.cigarette.resize(
            (face_width, int(face_width * self.cigarette.size[1]/self.cigarette.size[0])),
            resample = lmage.LANCZOS )

        x = mouth[0,0] - face_width + int(16 * face_width / self.cigarette.size[0])
        y = mouth_center[1]
        return {"image":cigarette,"pos":(x,y)}

    def oriemtation(self, rects, img_gray):
        """
        人脸定位
        :param rects:
        :param img_gray:
        :return:
        """

        faces = []
        for rect in rects:
            face ={}
            face_shades_width = rect.right() - rect.left()
            predictor_shape = self.predictor(img_gray. rect)
            face_shape = face_utils.shape_to_np(predictor_shape)
            face['cigarette'] = self.get_cigarette_info(face_shape, face_shape_width)
            face['glasses'] = self.get_glasses_info(face_shape,face_shades_width)

            faces.append(face)
        return faces

    def listener_keys(self):
        """
        刚才我们提到了键盘监听事件，这里我们实现一下这个函数：
        设置键盘监听事件
        :return:
        """
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.listener = False
            self.console("程序退出")
            sleep(1)
            self.exit()

        if key == ord("d"):
            self.doing = not self.doing

    def init_mask(self):
        """
        接下来我们实现加载面具信息的函数：
        :return:
        """
        self.console("加载面具...")
        self.deal, self.text, self.cigarette =(
            lmage.open(x) for x in ["images/deals.png","images/text.png","images/cigearette.png"] )

    def drawing(self, draw_img, faces):
        """
        上面基本的功能都实现了，我们该实现画图函数了
        画图
        :param draw_img:
        :param faces:
        :return:
        """
        for face in faces:
            if self.animation_time<self.duration - self.fixed_time:
                current_x = int(face["glasses"]["pos"][0])
                current_y = int(face["glasses"]["pos"][1] * self.animation_time / (self.duration - self.fixed_time))
                draw_img.paste(face["glasses"]["image"],(current_x),(current_y),face["glasses"]["image"])

                cigarette_x = int(face["cigarette"]["pos"][0])
                cigarette_y = int(face["cigarette"]["pos"][1] * self.animation_time / (self.duration - self.fixed_time))
                draw_img.paste(face["cigarette"]["image"],(current_x),(current_y),face["cigarette"]["image"])
            else:
                draw_img.paste(face["glasses"]["image"],face["glasses"]["pos"],["glasses"]["image"])
                draw_img.paste(face["cigarette"]["image"],face["cigarette"]["pos"],["cigarette"]["image"])
                draw_img.paste(self.text,(75,draw_img.height // 2 + 128),self.text)

        """既然是一个服务类， 那该有启动与退出函数"""
        """
        start()函数，启动后根据初始化监听信息，不断监听视频流，并将流信息通过opencv转换成图像展示出来并调用按键监听函数，不断的
        监听你是否按下"d"键进行面具加载，如果监听成功，则进行图像人脸检测，并移动面具，并持续一个周期的时间结束，面具此时会根据
        你的面部移动而移动。最终呈现文档顶部的效果
        """

    def start(self):
        """
        启动程序
        :return:
        """
        self.console("程序启动成功")
        self.init_mask()
        while self.listener:
            frame = self.read_data()
            frame = resize(frame,width=self.max_width)
            img_grat = cv2.cvtColor(frame.cv2.COLOR_BGR2GRAY)
            rects = self.detector(img_grat,0)
            faces = self.oriemtation(rects,img_grat)
            draw_img = lmage.fromarray(cv2.cvtColor(frame,cv2.COLOR_BG2BGR))
            if self.doing:
                self.drawing(draw_img,faces)
                self.animation_time += self.speed
                self.save_data(draw_img)
                if self.animation_time > self.duration:
                    self.doing = False
                    self.animation_time = 0
                else:
                    frame = cv2.cvtColor(np.asarray(draw_img),cv2.COLOR_BG2BGR)
            cv2.imshow("hello mask",frame)
            self.listener_keys()

    def exit(self):
        """
        程序退出
        :return:
        """
        self.video_capture.release()
        cv2.destroyAllWindows()

    """启动"""
    if __name__ == '__main__':
        ms = DynamicStreamMaskService()
        ms.start()