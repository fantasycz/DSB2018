import os, sys
sys.path.append(os.path.dirname(__file__))

from train_0 import *

## overwrite functions ###
def revert(net, images):

    def torch_clip_proposals (proposals, index, width, height):
        boxes = torch.stack((
             proposals[index,0],
             proposals[index,1].clamp(0, width  - 1),
             proposals[index,2].clamp(0, height - 1),
             proposals[index,3].clamp(0, width  - 1),
             proposals[index,4].clamp(0, height - 1),
             proposals[index,5],
             proposals[index,6],
        ), 1)
        return proposals

    # ----

    batch_size = len(images)
    for b in range(batch_size):
        image  = images[b]
        height,width  = image.shape[:2]

        index = (net.detections[:,0]==b).nonzero().view(-1)
        net.detections   = torch_clip_proposals (net.detections, index, width, height)

        net.masks[b] = net.masks[b][:height,:width]

    return net, image

def eval_augment(image, multi_mask, meta, index):
    
    image = scale_image_canals(image)
    pad_image = pad_to_factor(image, factor=16)
    input = torch.from_numpy(pad_image.transpose((2,0,1))).float().div(255)
    box, label, instance  = multi_mask_to_annotation(multi_mask)

    return input, box, label, instance, meta, image, index


def eval_collate(batch):

    batch_size = len(batch)
    #for b in range(batch_size): print (batch[b][0].size())
    inputs    = torch.stack([batch[b][0]for b in range(batch_size)], 0)
    boxes     =             [batch[b][1]for b in range(batch_size)]
    labels    =             [batch[b][2]for b in range(batch_size)]
    instances =             [batch[b][3]for b in range(batch_size)]
    metas     =             [batch[b][4]for b in range(batch_size)]
    images    =             [batch[b][5]for b in range(batch_size)]
    indices   =             [batch[b][6]for b in range(batch_size)]

    return [inputs, boxes, labels, instances, metas, images, indices]


def average_iou_meric(cfg, image, mask, truth_box, truth_label, truth_instance):

    H,W = image.shape[:2]
    average_overlap   = 0
    average_precision = 0

    if len(truth_box)>0:

        #pixel error: fp and miss
        truth_mask = instance_to_multi_mask(truth_instance)
        truth      = truth_mask!=0

        predict = mask!=0
        hit  = truth & predict
        miss = truth & (~predict)
        fp   = (~truth) & predict

        # metric -----
        predict = mask
        truth   = instance_to_multi_mask(truth_instance)

        num_truth   = len(np.unique(truth  ))-1
        num_predict = len(np.unique(predict))-1

        if num_predict!=0:
            intersection = np.histogram2d(truth.flatten(), predict.flatten(), bins=(num_truth+1, num_predict+1))[0]

            # Compute areas (needed for finding the union between all objects)
            area_true = np.histogram(truth,   bins = num_truth  +1)[0]
            area_pred = np.histogram(predict, bins = num_predict+1)[0]
            area_true = np.expand_dims(area_true, -1)
            area_pred = np.expand_dims(area_pred,  0)
            union = area_true + area_pred - intersection
            intersection = intersection[1:,1:]   # Exclude background from the analysis
            union = union[1:,1:]
            union[union == 0] = 1e-9
            iou = intersection / union # Compute the intersection over union

            precision = {}
            average_precision = 0
            thresholds = np.arange(0.5, 1.0, 0.05)
            for t in thresholds:
                tp, fp, fn = compute_precision(t, iou)
                prec = tp / (tp + fp + fn)
                precision[round(t,2) ]=prec
                average_precision += prec
            average_precision /= len(thresholds)


            #iou = num_truth, num_predict
            overlap = np.max(iou,1)

            average_overlap = overlap.mean()

    return average_overlap



def run_generate_balancing_train_weights():
    out_dir  = RESULTS_DIR + '/config'
    initial_checkpoint = RESULTS_DIR + '/config/checkpoint/15020_model.pth'
        
    test_dataset = ScienceDataset('train_ids_fix_598', mode='train',transform = eval_augment)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler = SequentialSampler(test_dataset),
                        batch_size  = 1,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True,
                        collate_fn  = eval_collate)
    
    cfg = Configuration()
    net = MaskRcnnNet(cfg).cuda()
    if initial_checkpoint is not None:
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
    
    results = dict()
    for i, (inputs, truth_boxes, truth_labels, truth_instances, metas, images, indices) in enumerate(test_loader, 0):
        if all((truth_label>0).sum()==0 for truth_label in truth_labels): continue

        net.set_mode('test')
        with torch.no_grad():
            inputs = Variable(inputs).cuda()
            net(inputs, truth_boxes,  truth_labels, truth_instances )
        
        ##save results ---------------------------------------
        revert(net, images)


        batch_size = len(indices)
        assert(batch_size==1)  #note current version support batch_size==1 for variable size input
                               #to use batch_size>1, need to fix code for net.windows, etc

        batch_size,C,H,W = inputs.size()
        inputs = inputs.data.cpu().numpy()

        masks      = net.masks
        detections = net.detections.cpu().numpy()


        for b in range(batch_size):
            image  = images[b]
            height,width  = image.shape[:2]
            mask = masks[b]

            index = np.where(detections[:,0]==b)[0]
            detection = detections[index]
            box = detection[:,1:5]


            truth_mask = instance_to_multi_mask(truth_instances[b])
            truth_box  = truth_boxes[b]
            truth_label= truth_labels[b]
            truth_instance= truth_instances[b]

            '''
            mask_average_precision, mask_precision =\
                compute_average_precision_for_mask(mask, truth_mask, t_range=np.arange(0.5, 1.0, 0.05))

            box_precision, box_recall, box_result, truth_box_result = \
                compute_precision_for_box(box, truth_box, truth_label, threshold=[0.5])
            box_precision = box_precision[0]

            mask_average_precisions.append(mask_average_precision)
            box_precisions_50.append(box_precision)
            '''
            
            average_iou = average_iou_meric(cfg, image, mask, truth_box, truth_label, truth_instance)

            # --------------------------------------------
            id_ = test_dataset.ids[indices[b]]
            print('%d\t %s\t %0.5f'%(i, id_,average_iou))
            results[id_] = [average_iou]

        with open(out_dir+'/weights.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in results.items():
                writer.writerow([key, value])
        '''
        with open('weights.csv', 'r') as csv_file:
            reader = csv.reader(csv_file)
            mydict = dict(reader)
        '''
            
            
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    run_generate_balancing_train_weights()



    print('\nsucess!')