from ocr_engine import get_engine

print("1.开始获取引擎")
engine=get_engine()

print("2.获取引擎成功")
image_path="/home/lxr/projects/deepseek_ocr_demo/goodluck.jpeg"
print("3.开始识别：",image_path)

text=engine.ocr_image(image_path)

print("4.识别完成")

print("repr = ",repr(text))
print("text = ",text)
