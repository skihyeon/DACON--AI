from PIL import Image
import pandas as pd
import os

def split_image(img_path, split_ratio, molding_path, leadframe_path):
    img = Image.open('./datas/' + img_path)
    w, h = img.size

    split_point = int(h*split_ratio)

    top = img.crop((0,0, w, split_point))
    bot = img.crop((0,split_point,w,h))

    top.save(molding_path+'/'+img_path.split('/')[-1].split('.')[0]+'_molding.png')
    bot.save(leadframe_path+'/'+img_path.split('/')[-1].split('.')[0]+'_leadframe.png')

def list_png_images(dir, train=True, labels = None):
    files = os.listdir(dir)
    idx = [num.split('.')[0] for num in files]
    img_path = [dir + '/' + num for num in files]

    result_dict = {
                    'id': idx,
                    'img_path': img_path
                    }
    result_df = pd.DataFrame(result_dict)

    if train == True:
        result_df['label'] = 0
    
    if train == False:
        result_df['label'] = labels

    return result_df
    

if __name__=="__main__":
    molding_path = './datas/train/moding'
    leadframe_path = './datas/train/leadframe'
    train_data_path = list(pd.read_csv('./datas/train.csv')['img_path'])

    os.makedirs(molding_path, exist_ok=True)
    os.makedirs(leadframe_path, exist_ok=True)

    for i in train_data_path:
        split_image(i, 0.5, molding_path, leadframe_path)

    train_molding = list_png_images(molding_path)
    train_molding.to_csv('./datas/train_molding.csv', index=False)

    train_leadframe = list_png_images(leadframe_path)
    train_leadframe.to_csv('./datas/train_leadframe.csv', index=False)

    test_molding_path = './datas/test/molding'
    test_leadframe_path = './datas/test/leadframe'
    test_data_path = list(pd.read_csv('./datas/test.csv')['img_path'])

    os.makedirs(test_molding_path, exist_ok=True)
    os.makedirs(test_leadframe_path, exist_ok=True)

    test_labels = pd.read_csv('./datas/test.csv')['label']
    for i in test_data_path:
        split_image(i, 0.5, test_molding_path, test_leadframe_path)

    test_molding = list_png_images(test_molding_path,  train=False, labels= test_labels)
    test_molding.to_csv('./datas/test_molding.csv', index=False)

    test_leadframe = list_png_images(test_leadframe_path,  train=False, labels= test_labels)
    test_leadframe.to_csv('./datas/test_leadframe.csv', index=False)
