from common import *
from utility.file import *
from utility.draw import *

from dataset.reader import *

def run_make_test_annotation():

    split = 'test_ids_65'
    ids = read_list_from_file(DATA_DIR + '/split/' + split, comment='#')

    for id_ in ids:
        image_file = DATA_DIR + '/stage1_test/%s/images/%s.png'%(id_, id_)

        #image
        image = cv2.imread(image_file,cv2.IMREAD_COLOR)

        ## save and show -------------------------------------------
        # image_show('image',image)

        cv2.imwrite(DATA_DIR + '/zhen_fix/stage1_test/images/%s.png'%(id_),image)
        # cv2.waitKey(1)



def run_make_train_annotation():
    
    split = 'train_ids_664'
    ids = read_list_from_file(DATA_DIR + '/split/' + split, comment='#')

    data_dir = DATA_DIR + '/zhen_fix/stage1_train'
    os.makedirs(data_dir + '/multi_masks', exist_ok=True)
    os.makedirs(data_dir + '/overlays', exist_ok=True)
    os.makedirs(data_dir + '/images', exist_ok=True)

    for id_ in ids:
        image_files = glob.glob(DATA_DIR + '/kaggle-dsbowl-2018-dataset-fixes-master/stage1_train/%s/images/*.png'%(id_))
        assert(len(image_files)==1)
        image_file=image_files[0]
        print(id_)

        #image
        image = cv2.imread(image_file,cv2.IMREAD_COLOR)[:,:,:3]

        H,W,C      = image.shape
        multi_mask = np.zeros((H,W), np.int32)

        mask_files = glob.glob(DATA_DIR + '/kaggle-dsbowl-2018-dataset-fixes-master/stage1_train/%s/masks/*.png'%(id_))
        mask_files.sort()

        for i, mask_file in enumerate(mask_files):
            mask = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE)
            multi_mask[np.where(mask>128)] = i+1

        #check
        color_overlay   = multi_mask_to_color_overlay  (multi_mask,color='summer')
        color1_overlay  = multi_mask_to_contour_overlay(multi_mask,color_overlay,[255,255,255])
        contour_overlay = multi_mask_to_contour_overlay(multi_mask,image,[0,255,0])
        all = np.hstack((image, contour_overlay,color1_overlay,)).astype(np.uint8)

        # cv2.imwrite(data_dir +'/images/%s.png'%(name),image)


        np.save(data_dir + '/multi_masks/%s.npy' % id_, multi_mask)
        cv2.imwrite(data_dir + '/multi_masks/%s.png' % id_, color_overlay)
        cv2.imwrite(data_dir + '/overlays/%s.png' % id_, all)
        cv2.imwrite(data_dir + '/images/%s.png' % id_, image)


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_make_train_annotation()
    print('Train annotation is finished!')
    run_make_test_annotation()
    print('Test annotation is finished!')
    print( 'sucess!')


