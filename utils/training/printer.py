def print_epoch(model_type,epoch,time,losses,AUC):
      print ('__________________')
      print('Epoch {} at {} sec \n{} losses: {} \nAUC = {}'.format(epoch,
                                                                   time,
                                                                   model_type,
                                                                   losses,
                                                                   AUC))

