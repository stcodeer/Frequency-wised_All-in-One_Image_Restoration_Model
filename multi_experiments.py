import os

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

def experiment(hyperparams):
    try:
        print('%s' % ('training: h ' + hyperparams))
        os.system('python train.py %s --output_path output/h%s/'%(hyperparams, tostr(hyperparams)))
        print('%s' % ('testing: h ' + hyperparams))
        os.system('python test.py %s --output_path output/h%s/'%(hyperparams, tostr(hyperparams)))
        
    except Exception as e:
        log_file.write('[Error] %s\n' % e)
        
    else:
        log_file.write('[Done]\n')
        
    log_file.write('%s\n' % ('training: h' + hyperparams))
    log_file.write('%s\n' % ('testing: h' + hyperparams))

    log_file.flush()


hyperparams = '--epochs 1'

experiment(hyperparams)

log_file.close()