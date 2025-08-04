from mtcnn import MTCNN
import cv2
import os

if __name__ == '__main__':
    images = os.listdir('File_data')
    detector = MTCNN()
    os.chdir('File_data')
    for image in images:
        name_root = image.split('/')[-1][:-4]
        if image.split('/')[-1][-3:] != 'mp4':
            img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(img)
        if len(results) > 0:
            i = 1
            for result in results:
                if result['confidence'] > 0.9:
                    x = int(result['box'][0])
                    y = int(result['box'][1])
                    width = int(result['box'][2])
                    height = int(result['box'][3])
                    face = cv2.cvtColor(img[y:y + height, x:x + width], cv2.COLOR_RGB2BGR)
                    name_save = name_root + '_' + str(i) + '.jpg'
                    root = 'C:/Users/Buu/PycharmProjects/Web_vnsf/Faces'
                    path = os.path.join(root,name_save)
                    cv2.imwrite(path, face)
                    i = i + 1