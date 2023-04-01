from skimage import io,data,exposure
import cv2
import matplotlib.pyplot as plt
moon=data.moon()
moon2=cv2.equalizeHist(moon)
moon3=exposure.equalize_hist(moon)
# plt.imshow(moon2,cmap="gray")
# plt.show()
cv2.imshow('moon',moon)
cv2.waitKey(0)
cv2.imshow('moon',moon3)
cv2.waitKey(0)
cv2.destroyAllWindows()