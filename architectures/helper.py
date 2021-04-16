from utils import save_training_curves
from utils.plotting import save_training_curves
from utils.metrics import get_classifcation, get_nln_metrics, save_metrics, accuracy_metrics

def end_routine(train_images, test_images, test_labels, test_masks, model, model_type, args):
    save_training_curves(model,args,test_images,test_labels,model_type)
    auc_latent, f1_latent, neighbour,radius = get_nln_metrics(model,
                                                              train_images,
                                                              test_images,
                                                              test_labels,
                                                              model_type,
                                                              args)
    
    auc_recon ,f1_recon = get_classifcation(model_type,
                                            model,
                                            test_images,
                                            test_labels,
                                            args,
                                            f1=True)
    save_metrics(model_type,
                 args,
                 auc_recon, 
                 f1_recon,
                 neighbour,
                 radius,
                 auc_latent,
                 f1_latent)

    if args.data == 'MVTEC':
        accuracy_metrics(model,
                         train_images,
                         test_images,
                         test_labels,
                         test_masks,
                         model_type,
                         args)

