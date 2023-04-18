import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('C:\\Users\\haoyu\\Desktop\\lena.jpg',0)
img_float32 = np.float32 (img)
dft=cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)#低频转换到中间位置
#得到灰度图能表示的形式
magnitude_spectrum = 20*np.log(cv2.magnitude (dft_shift[:, :, 0], dft_shift[:,:, 1]))#对两个通道进行转换
plt.subplot(121),plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt. subplot(122), plt.imshow(magnitude_spectrum, cmap= 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks
plt.show()#越离中间点近频率越低
