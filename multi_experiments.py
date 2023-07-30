import os

log_path = 'multi_experiments.log'
log_file = open(log_path, "w")

def tostr(hyperparams):
    hyperparams = hyperparams.split().split('--')
    str = ''
    for substr in hyperparams:
        str = str + '_' + substr
    return str

def experiment(hyperparams):
    try:
        print('%s\n' % ('exp_name: ' + hyperparams))
        os.system(r'python ./train.py %s --output_path output/%s/ --ckpt_path ckpt/%s/'%(hyperparams, tostr(hyperparams), tostr(hyperparams)))
        
    except Exception as e:
        log_file.write('[Error] %s\n' % e)
        
    else:
        log_file.write('[Done]\n')
        
    log_file.write('%s\n' % ('exp_name: ' + hyperparams))

    log_file.flush()


hyperparams = ''

experiment()

log_file.close()