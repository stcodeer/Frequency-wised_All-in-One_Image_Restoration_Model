import os

if not os.path.exists('output'):
        os.mkdir('output')
        
log_path = 'multi_experiments.log'
log_file = open(log_path, "w")

def tostr(hyperparams):
    if len(hyperparams) == 0:
        return ''
    
    hyperparams = hyperparams.split(' ')
    str = ''
    for substr in hyperparams:
        if len(substr) > 0:
            str = str + '_' + substr.replace('--', '')
    return str

def experiment(hyperparams, train, test):
    try:
        if train:
            print('%s' % ('training: h ' + hyperparams))
            os.system('python train.py %s --output_path output/h%s/'%(hyperparams, tostr(hyperparams)))
        if test:
            print('%s' % ('testing: h ' + hyperparams))
            os.system('python test.py %s --output_path output/h%s/'%(hyperparams, tostr(hyperparams)))
        
    except Exception as e:
        log_file.write('[Error] %s\n' % e)
        
    else:
        log_file.write('[Done]\n')
    
    if train:
        log_file.write('%s\n' % ('training: h' + hyperparams))
    if test:
        log_file.write('%s\n' % ('testing: h' + hyperparams))

    log_file.flush()

train = True
test = True

hyperparams0 = "--epochs_encoder 0 --epochs 2 --crop_test_imgs_size 128"
experiment(hyperparams0, train, test)

# hyperparams1 = "--crop_test_imgs_size 512 --de_type denoising_15 denoising_25 denoising_50 deraining dehazing --test_de_type denoising_15 denoising_25 denoising_50 deraining dehazing "
# experiment(hyperparams1, train, test)

# hyperparams2 = "--crop_test_imgs_size 128"
# experiment(hyperparams2, train, test)

# hyperparams3 = "--encoder_type ViT --crop_test_imgs_size 128"
# experiment(hyperparams3, train, test)

# hyperparams4 = "--crop_test_imgs_size 512"
# experiment(hyperparams4, train, test)


log_file.close()