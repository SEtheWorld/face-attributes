import tqdm
import insightface
import glob
import pandas as pd
import cv2
import numpy as np

if __name__ == '__main__':
    model = insightface.model_zoo.get_model('retinaface_r50_v1')
    model.prepare(ctx_id = 0, nms=0.4)

    csv_path = '/home/Data/appa-real/processed/'

    for file in glob.glob(csv_path + '/*.csv'):
        df = pd.read_csv(file)
        crop_path = []
        for i in tqdm.tqdm(range(len(df))):
            try:
                img = cv2.imread(df['file_name'][i])
                img = cv2.resize(img, (512, 512)) #Resize all images to the same size (512, 512, 3)
                bbox, landmark = model.detect(img, threshold=0.5, scale=1.0)

                #Get bbox and landmark of face which has the biggest area
                area = (bbox[:, 2] - bbox[:, 0])*(bbox[:, 3] - bbox[:, 1])
                choose_idx = np.argmax(area, axis=-1).ravel()
                choose_idx = int(choose_idx)    
                
                landmark_new = np.reshape(landmark, (-1, 10), order='F')
                landmark_new = landmark_new.astype('int')

                x1 = int(bbox[choose_idx][0])
                y1 = int(bbox[choose_idx][1])
                x2 = int(bbox[choose_idx][2]) 
                y2 = int(bbox[choose_idx][3])

                crop_img = img[y1 : y2, x1 : x2]
                crop_img = cv2.resize(crop_img, (224, 224))
                
                cv2.imwrite(str(df['file_name'][i]) + '_crop.jpg', crop_img)
                crop_path.append(str(df['file_name'][i]) + '_crop.jpg')
            except Exception:
                crop_path.append(str(df['file_name'][i]))
                continue
        
        df['crop_path'] = crop_path
        df.to_csv(file, index=False, header=True)
