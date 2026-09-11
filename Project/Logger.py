import os

class Logger:
    def __init__(self):
        super(Logger, self).__init__()

    def open(self, name, mode):
        self.txt= open(name, mode=mode)

    def write(self, str_):
        self.txt.write(str_)
        print(str_)

    def close(self):
        self.txt.close()

def create_logger(mode, model_name='resnet-50', round_=None, data_mode='Normal'):
    out_dir= './{}_{}_weight_file/{}/'.format(mode, model_name, data_mode, round_)
    os.makedirs(out_dir+'/checkpoint/', exist_ok=True)
    os.makedirs(out_dir+'/train/', exist_ok=True)

    log= Logger()
    log.open(out_dir+'/log.single_modality_{}_skinlesion.txt'.format(mode), mode='w')
    log.write('\n--- [START %s] %s\n\n' % ('IDENTIFIER', '-'*64))
    log.write('\tout_dir= %s\n' % out_dir)
    log.write('\n')
    log.write('\t<additional comments> ... \n')
    log.write('\t - {} {} \n'.format(mode, model_name))
    log.write('\t - simple augmentation \n')
    log.write('\n')
    return log, out_dir

