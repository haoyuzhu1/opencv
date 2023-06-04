# 导入工具包
import numpy as np
import argparse
import imutils
import cv2

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# 正确答案
# 字典：键值对，第几个位置对应正确答案对应的索引值
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

# 提取轮廓


def order_points(pts):
	# 一共4个坐标点，定义一个4行两列的元素为0的矩阵。
	rect = np.zeros((4, 2), dtype="float32")
	# 找出图像中四个顶点的坐标。
	# 按顺序找到对应坐标0123分别是 左下、右下、右上、左上
	# 计算左上，右下
	# 现在pts中是4个向量，axis为1表示将矩阵的行向量相加。
	# [[0, 1],
	#  [1, 1],
	#  [1, 0],
	#  [0, 0]]
	s = pts.sum(axis=1)
	print('s', s)
	# rect[0]表示左下， rect[2]表示右上
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	print('rect[0][2]', rect[0], rect[2])
	# 计算右下和左上
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	print('rect[1][3]', rect[1], rect[3])
	return rect
# 执行完order_point操作后，pts中存放的是按照左下、右下、右上、左上顺序排列的四个顶点的坐标


def four_point_transform(image, pts):
	# 获取输入坐标点
	rect = order_points(pts)
	(bl, br, tr, tl) = rect

	# 计算输入的w和h值
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# 变换后对应坐标位置
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype="float32")

	# 计算变换矩阵
	# 透视变换
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# 返回变换后结果
	return warped

# 对检测到的轮廓进行排序


def sort_contours(cnts, method="left-to-right"):
	reverse = False
	i = 0
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
								key=lambda b: b[1][i], reverse=reverse))
	return cnts, boundingBoxes


def cv_show(name,img):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# 预处理


image = cv2.imread(args["image"])
contours_img = image.copy()
# 彩色图片转化为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 高斯滤波去噪
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv_show('GaussianBlured',blurred)
# Canny边缘检测
edged = cv2.Canny(blurred, 75, 200)
cv_show('Canny',edged)

# 用边缘检测的结果进行轮廓检测
# cv2.RETR_EXTERNAL表示只检测最外层的轮廓
# cv2.findContours函数有三个返回值：
# binary（二值图像）、contours（List存放检测到的所有轮廓）、hierarchy（表示层级信息）
# 查找的轮廓保存在cnts中
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 画出轮廓
# cv2.drawContours(image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None)
# 第一个参数是指明在哪幅图像上绘制轮廓；image为三通道才能显示轮廓
# 第二个参数是轮廓本身，在Python中是一个list;
# 第三个参数指定绘制轮廓list中的哪条轮廓，如果是-1，则绘制其中的所有轮廓。后面的参数很简单。其中thickness表明轮廓线的宽度，如果是-1（cv2.FILLED），则为填充模式。
cnts = cnts[1] if imutils.is_cv3() else cnts[0]
cv2.drawContours(contours_img, cnts, -1, (0, 0, 255), 3)

cv_show('contours_img',contours_img)
docCnt = None

# 确保检测到了轮廓
if len(cnts) > 0:
	# 根据轮廓大小进行排序
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

	# 遍历每一个轮廓
	for c in cnts:
		# 近似
		# cv2.arcLength函数返回值是轮廓的周长，第二个参数表示轮廓是否闭合
		peri = cv2.arcLength(c, True)
		# cv2.approxPolyDP函数对轮廓进行近似处理，在一定精度范文诶，将弧用弦来近似代替。
		# 通常精度用周长的百分比来计算
		# 返回值是围成最大轮廓的点的坐标
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		print('approx', approx)
		# 准备做透视变换
		if len(approx) == 4:
			docCnt = approx
			break

# 执行透视变换

warped = four_point_transform(gray, docCnt.reshape(4, 2))
cv_show('the result of four point transform',warped)
# Otsu's 阈值处理
# 大津法进行图像二值化处理，对于双峰的情况能够自动识别出阈值
# 使用自适应的二值化处理
thresh = cv2.threshold(warped, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] 
cv_show('thresh',thresh)
thresh_Contours = thresh.copy()
# 找到每一个圆圈轮廓
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[1] if imutils.is_cv3() else cnts[0]
cv2.drawContours(thresh_Contours, cnts, -1, (0, 0, 255), 3)
cv_show('thresh_Contours',thresh_Contours)
questionCnts = []

# 遍历
for c in cnts:
	# 遍历轮廓，计算轮廓的外接矩形，满足固定比例则看做答题的圆圈位置。
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)
	# 根据实际情况指定标准
	# 宽度和高度都大于20，而且宽高比在0.9和1.1之间。
	# 则判定该轮廓为答题的圆圈
	if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
		questionCnts.append(c)

# 按照从左到右从上到下进行排序
questionCnts = sort_contours(questionCnts,
	method="top-to-bottom")[0]
# 初始化的得分为0
correct = 0

# 每排有5个选项
# 从0到len(questionCnts)
# 枚举，分别取0， 5， 10，……
# 遍历不同的题目
for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
	# 排序
	cnts = sort_contours(questionCnts[i:i + 5])[0]
	print('cnts', cnts)
	bubbled = None

	# 遍历每一个结果
	# 对于统一体的不同选项
	for (j, c) in enumerate(cnts):
		# 使用mask来判断结果
		# 制作每一个选项的圆圈mask，只有选项的位置为255，其他位置为0。
		mask = np.zeros(thresh.shape, dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1) #-1表示填充
		cv_show('mask',mask)
		# 通过计算非零点数量来算是否选择
		# 将mask于二值化的图像进行与运算
		# 利用掩膜（mask）进行“与”操作，
		# 即掩膜图像白色区域是对需要处理图像像素的保留，
		# 黑色区域是对需要处理图像像素的剔除，
		# 其余按位操作原理类似只是效果不同而已。
		# 本实例中依次保留每一个选项的圆圈区域。
		mask = cv2.bitwise_and(thresh, thresh, mask=mask)
		# cv2.countNonZero函数的作用是统计非零像素点。
		# 也就是判断这个选项涂没涂上。
		total = cv2.countNonZero(mask)

		# 通过阈值判断
		if bubbled is None or total > bubbled[0]:
			bubbled = (total, j)

	# 对比正确答案
	color = (0, 0, 255)
	k = ANSWER_KEY[q]

	# 判断正确
	if k == bubbled[1]:
		color = (0, 255, 0)
		correct += 1

	# 绘图
	cv2.drawContours(warped, [cnts[k]], -1, color, 3)

# 计算得分
score = (correct / 5.0) * 100
print("[INFO] score: {:.2f}%".format(score))
# 把得分以文本的形式显示在图片上。
cv2.putText(warped, "{:.2f}%".format(score), (10, 30),
	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", warped)
cv2.waitKey(0)

