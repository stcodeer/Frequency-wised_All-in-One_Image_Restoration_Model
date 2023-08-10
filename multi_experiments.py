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
        
            # for epoch in range(100, 1100, 100):
            #     os.system('python test.py %s --epochs %s --output_path output/h%s/'%(hyperparams, str(epoch), tostr(hyperparams)))
                
            # os.system('python plot_performance_curve.py %s --output_path output/h%s/'%(hyperparams, tostr(hyperparams)))
            # os.system('python plot_lamb_curve.py %s --output_path output/h%s/'%(hyperparams, tostr(hyperparams)))
    except Exception as e:
        log_file.write('[Error] %s\n' % e)
        
    else:
        log_file.write('[Done]\n')
    
    if train:
        log_file.write('%s\n' % ('training: h' + hyperparams))
    if test:
        log_file.write('%s\n' % ('testing: h' + hyperparams))

    log_file.flush()

train = False
test = True

hyperparams = "--cuda 0 --encoder_type ViT --batch_wise_decompose True --frequency_decompose_type 5_bands --crop_test_imgs_size 128 --de_type denoising_15 denoising_25 denoising_50 deraining --test_de_type denoising_bsd68_15 denoising_bsd68_25 denoising_bsd68_50 deraining"
experiment(hyperparams, train, test)


log_file.close()