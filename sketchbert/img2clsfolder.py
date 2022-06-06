import os
import matplotlib.pyplot as plt

f = open("char_to_idx.txt", encoding="utf-8")
cls = f.read()
f.close()

path = "oracle_3shot"
imgnames = os.listdir(path)

for imgname in imgnames:
    if imgname.endswith('png'):
        name_list = imgname.split("_")
        id = int(name_list[3])
        index = id // 3
        prob = name_list[4]
        print(id, prob)
        savename = str(id) + "_" + prob

        savepath = os.path.join('data_3shot', cls[index])
        os.makedirs(savepath, exist_ok=True)

        img = plt.imread(os.path.join(path, imgname))
        plt.imsave(os.path.join(savepath, savename), img)