# Import the necessary Libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np

def traslation(img):

    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    width = image_rgb.shape[1]
    height = image_rgb.shape[0]

    tx = 200
    ty = 200


    translation_matrix = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)

    translated_image = cv2.warpAffine(image_rgb, translation_matrix, (width, height))


    fig, axs = plt.subplots(1, 2, figsize=(7, 4))


    axs[0].imshow(image_rgb)
    axs[0].set_title('Original Image')


    axs[1].imshow(translated_image)
    axs[1].set_title('Image Translation')


    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])


    plt.tight_layout()
    plt.show()
def rotation(image):
    gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow('grayscale',gray_image)
    center = (gray_image.shape[1] // 2, gray_image.shape[0] // 2)
    angle = 60
    scale = 2
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv2.warpAffine(gray_image, rotation_matrix, (image.shape[1], image.shape[0]))
    fig, axs = plt.subplots(1, 2, figsize=(7, 4))


    axs[0].imshow(gray_image)
    axs[0].set_title('Original Image')
    

    axs[1].imshow(rotated_image)
    axs[1].set_title('Image Rotation')


    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])


    plt.tight_layout()
    plt.show()
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
def grey(image):
    gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow('grayscale',gray_image)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
if __name__ =="__main__":
    img = cv2.imread("website.png")
    traslation(img)
    grey(img)
    rotation(img)
