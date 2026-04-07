"""
人脸检测
"""
import cv2
import os

# 方法1：使用Anaconda自带的分类器
anaconda_cascade_path = "D:/Anaconda3/envs/opencv_env/Library/etc/haarcascades/haarcascade_frontalface_default.xml"

# 检查Anaconda环境中的文件是否存在
if os.path.exists(anaconda_cascade_path):
    face_cascade = cv2.CascadeClassifier(anaconda_cascade_path)
    print("使用Anaconda自带的分类器")
else:
    # 方法2：使用相对路径
    relative_path = "./haarcascades/haarcascade_frontalface_default.xml"
    if os.path.exists(relative_path):
        face_cascade = cv2.CascadeClassifier(relative_path)
        print("使用相对路径的分类器")
    else:
        print("错误: 找不到分类器文件！")
        print("请确保haarcascades文件夹和XML文件存在")
        exit()

# 检查分类器是否加载成功
if face_cascade.empty():
    print("错误: 分类器加载失败！")
    exit()

# 读取图片
img = cv2.imread("C:/Users/CGR/Pictures/Beauty.jpg")

if img is None:
    print("错误: 无法读取图像文件！")
    print("请检查图像路径: C:/Users/CGR/Pictures/Beauty.jpg")
    exit()

# 转为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 人脸检测
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

print(f"检测到 {len(faces)} 张人脸")

# 绘制矩形框
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 显示结果
cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()