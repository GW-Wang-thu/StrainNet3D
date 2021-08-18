import cv2
import numpy as np
import os


def check_imgs(dataset_dir):
    allfiles = os.listdir(dataset_dir)
    Lrefimgs = [os.path.join(dataset_dir, f) for f in allfiles if f.endswith('LRLD_Imgs.npy')]

    for i in range(len(Lrefimgs)):
        print("\b", i, end="\r")
        img = np.load(Lrefimgs[i])[0, :, :]
        tem_id = Lrefimgs[i][-22:-13]
        frame = cv2.putText(img, tem_id, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
        cv2.imshow('frame', frame)
        cv2.imwrite(dataset_dir + "imgs/"+tem_id+".bmp", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    check_imgs("D:/Guowen/DLDIC_3D/dataset/Valid/")