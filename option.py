import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs to train the total model.')
parser.add_argument('--epochs_encoder', type=int, default=100, help='number of epochs to train encoder.')
parser.add_argument('--lr', type=float, default=None, help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', type=str, default=['denoising_15', 'denoising_25', 'denoising_50', 'deraining', 'dehazing', 'deblurring'],
                    help='which type of degradations are training for.')
parser.add_argument('--test_de_type', nargs='+', type=str, default=['denoising_bsd68_15', 'denoising_bsd68_25', 'denoising_bsd68_50', 'deraining', 'dehazing', 'deblurring'],
                    help='which type of degradations are testing for.')
# other available test type: 'denoising_urban100_15', 'denoising_urban100_25', 'denoising_urban100_50'

parser.add_argument('--patch_size', type=int, default=128, help='patcphsize of input.')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers.')

parser.add_argument('--save_imgs', type=bool, default=False, help='whether or not to save output images.')
parser.add_argument('--crop_test_imgs_size', type=int, default=512, help='crop test images to smaller than given resolution.')

# Path
parser.add_argument('--output_path', type=str, default='output/tmp/', help='output and checkpoint save path')

# Network Hyperparameters
parser.add_argument('--encoder_type', type=str, default='ResNet', help="should be in '[ResNet, ViT]'")
parser.add_argument('--encoder_dim', type=int, default=None, help='the dimensionality of encoder(default: 256 when ResNet, 3 when ViT).')


options = parser.parse_args()

options.batch_size = len(options.de_type)

options.ckpt_path = options.output_path + 'ckpt/'

if options.encoder_type == 'ResNet':
    
    if options.encoder_dim == None:
        options.encoder_dim = 256
        
    if options.lr == None:
        options.lr = 1e-3
        
elif options.encoder_type == 'ViT':
    
    if options.encoder_dim == None:
        options.encoder_dim = 3
        
    if options.lr == None:
        options.lr = 1e-4
else:
    assert False, "wrong encoder type."