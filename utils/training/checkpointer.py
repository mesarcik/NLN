import os
import tensorflow as tf

def save_checkpoint(model,epoch,args, model_type, model_subtype):
    dir_path = 'outputs/{}/{}/{}'.format(model_type,
                                          args.anomaly_class,
                                          args.model_name)
    if ((epoch + 1) % 10 == 0) :
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        model.save_weights('{}/training_checkpoints/checkpoint_{}'.format(dir_path,
                                                                     model_subtype))

    if (int(epoch)  == int(args.epochs)-1):
        model.save_weights('{}/training_checkpoints/checkpoint_full_model_{}'.format(
                                                                            dir_path,
                                                                    model_subtype))

        #TODO: This is a really ugly quick fix to write the config
        with open('{}/model.config'.format(dir_path), 'w') as fp:
            for arg in args.__dict__:
                fp.write('{}: {}\n'.format(arg, args.__dict__[arg]))

        print('Successfully Saved Model')

