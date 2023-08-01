import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs to train the total model.')
parser.add_argument('--epochs_encoder', type=int, default=100, help='number of epochs to train encoder.')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate of encoder.')

parser.add_argument('--de_type', type=list, default=['denoising_15', 'denoising_25', 'denoising_50', 'deraining', 'dehazing', 'deblurring'],
                    help='which type of degradations are training for.')
parser.add_argument('--test_de_type', type=list, default=['denoising_15', 'denoising_25', 'denoising_50', 'deraining', 'dehazing', 'deblurring'],
                    help='which type of degradations are testing for.')

parser.add_argument('--patch_size', type=int, default=128, help='patcphsize of input.')
parser.add_argument('--encoder_dim', type=int, default=256, help='the dimensionality of encoder.')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers.')

parser.add_argument('--save_imgs', type=bool, default=False, help='whether or not to save output images.')
parser.add_argument('--crop_test_imgs', type=bool, default=True, help='whether or not to crop test images to a resolution smaller than 512*512.')

# Path
parser.add_argument('--output_path', type=str, default='output/tmp/', help='output and checkpoint save path')


options = parser.parse_args()
options.batch_size = len(options.de_type)
options.ckpt_path = options.output_path + 'ckpt/'