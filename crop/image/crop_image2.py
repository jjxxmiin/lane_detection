from PIL import Image

img = Image.open('C:/0710pcv/test_image.png')
area = (100,300,700,500)
cropped_img = img.crop(area)

img.show()
cropped_img.show()