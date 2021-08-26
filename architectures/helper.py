from utils import save_training_curves
from utils.plotting import save_training_curves
from utils.metrics import get_classifcation, get_nln_metrics, save_metrics, accuracy_metrics

def end_routine(train_images, test_images, test_labels, test_masks, model, model_type, args):
    if model_type != 'DKNN':
        save_training_curves(model,args,test_images,test_labels,model_type)

    (auc_latent, neighbour,radius, dists_auc,sum_auc, mul_auc, x_hat, x_hat_train,
            neighbours_idx, neighbours_dist) = get_nln_metrics(model,   
                                                               train_images, 
                                                               test_images, 
                                                               test_labels, 
                                                               model_type, 
                                                               args)
    
    auc_recon = get_classifcation(model_type,
                                            model,
                                            test_images,
                                            test_labels,
                                            args)
    if False:
        seg_auc, seg_auc_nln, dists_auc, seg_dists_auc, seg_prc, seg_prc_nln, seg_iou, seg_iou_nln  = accuracy_metrics(model,
                                                                                                                        train_images,
                                                                                                                        test_images,
                                                                                                                        test_labels,
                                                                                                                        test_masks,
                                                                                                                        model_type,
                                                                                                                        neighbour,
                                                                                                                        radius,
                                                                                                                        args)
    else:
        seg_auc = 'n/a'
        seg_auc_nln = 'n/a'
        seg_dists_auc = 'n/a'
        seg_prc = 'n/a'
        seg_prc_nln = 'n/a' 
        seg_iou  = 'n/a'
        seg_iou_nln = 'n/a'


    save_metrics(model_type,
                 args,
                 auc_recon, 
                 seg_prc,
                 neighbour,
                 radius,
                 auc_latent,
                 seg_prc_nln,
                 seg_auc,
                 seg_auc_nln,
                 seg_iou,
                 seg_iou_nln,
                 dists_auc,
                 seg_dists_auc,
                 sum_auc,
                 mul_auc)


