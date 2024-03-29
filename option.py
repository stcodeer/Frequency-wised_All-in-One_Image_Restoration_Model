import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs to train the total model.')
parser.add_argument('--epochs_encoder', type=int, default=100, help='number of epochs to train encoder.')
parser.add_argument('--lr', type=float, default=None, help='learning rate of encoder.')
parser.add_argument('--contrast_loss_weight', type=float, default=None, help='contrast loss weight in objective function.')
parser.add_argument('--frequency_l1_loss_weight', type=float, default=0.1, help='frequency l1 loss weight in objective function.')

parser.add_argument('--de_type', nargs='+', type=str, default=['denoising_0', 'deraining', 'dehazing', 'deblurring'],
                    help='which type of degradations are training for.')
parser.add_argument('--test_de_type', nargs='+', type=str, default=['denoising_bsd68_15', 'denoising_bsd68_25', 'denoising_bsd68_50', 'deraining', 'dehazing', 'deblurring'],
                    help='which type of degradations are testing for.')
# other available test type: 'denoising_urban100_15', 'denoising_urban100_25', 'denoising_urban100_50'

parser.add_argument('--patch_size', type=int, default=128, help='patcphsize of input.')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')

parser.add_argument('--save_imgs', type=bool, default=False, help='whether or not to save output images.')
parser.add_argument('--crop_test_imgs_size', type=int, default=128, help='crop test images to smaller than given resolution.')

# Path
parser.add_argument('--output_path', type=str, default='output/tmp/', help='output and checkpoint save path')

# Network
parser.add_argument('--encoder_type', type=str, default='Uformer', help='should be in [ResNet, ViT, Uformer]')
parser.add_argument('--decoder_type', type=str, default='Uformer', help='should be in [ResNet, Uformer]')
parser.add_argument('--encoder_dim', type=int, default=None, help='the output dimensionality of encoder(default: 256 when ResNet, 3 when ViT, 256 when Uformer).')
parser.add_argument('--frequency_decompose_type', type=str, default='none',
                    help='should be in [%_bands, DC, none].(only available for ViT encoder and Uformer encoder+decoder)')

# Uformer encoder+decoder
parser.add_argument('--debug_mode', type=bool, default=False, help='whether or not to enable debug mode.(only available for Uformer encoder+decoder).')
parser.add_argument('--encoder_embed_dim', type=int, default=28, help='the embedding dimensionality of Uformer Encoder(only available for Uformer encoder+decoder).')
parser.add_argument('--embed_dim', type=int, default=56, help='the embedding dimensionality of Uformer Decoder(only available for Uformer encoder+decoder).')
parser.add_argument('--degradation_embedding_method', nargs='+', type=str, default=['residual'], 
                    help='degradation embedding method, should be in [residual, modulator, self_modulator, deform_conv, attention_residual, attention_kv, all_%_bands, all_DC].(only available for Uformer encoder+decoder).')
parser.add_argument('--learnable_modulator', type=bool, default=False, help='add learnable modulator in Uformer decoder.(only available for Uformer encoder+decoder)')
parser.add_argument('--num_frequency_bands_encoder', type=int, default=-1, help='the number of frequency bands used in encoder preprocess layer in the frequency domain.(only available for Uformer encoder+decoder)')
parser.add_argument('--num_frequency_bands', type=int, default=-1, help='the number of frequency bands used in contrast loss in the frequency domain.(only available for Uformer encoder+decoder)')
parser.add_argument('--num_frequency_bands_l1', type=int, default=-1, help='the number of frequency bands used in l1 loss in the frequency domain.(only available for Uformer encoder+decoder)')
parser.add_argument('--frequency_feature_enhancement_method', nargs='+', type=str, default=[], 
                    help='how feature maps used for calculating frequency domain losses are transformed to enhance contrast in the frequency domain, should be in [norm].(only available for Uformer encoder+decoder)')

parser.add_argument('--L', type=int, default=3, help='number of frequency bands used in attention map frequency modulation.(only available for Uformer encoder+decoder)')
parser.add_argument('--encoder_msa_type', type=str, default='freq', help='should be in [origin, freq].(only available for Uformer encoder+decoder)')

# ViT encoder
parser.add_argument('--out_channels', type=int, default=3, help='the hidden dimensionality of ViT encoder(only available for ViT encoder).')
parser.add_argument('--batch_wise_decompose', type=bool, default=False, help='use batch-wise learnable parameters for frequency decomposition module or not(only available for ViT encoder)')
parser.add_argument('--frequency_decompose_type_2', type=bool, default=False, help='(only available for ViT encoder)')

options = parser.parse_args()

if options.L == 3:
    contrast_loss_weight = 0.6
elif options.L == 2:
    contrast_loss_weight = 0.2
else:
    assert False

if options.de_type[0] == '2tasks':
    options.de_type = ['denoising_0', 'deraining']
    options.test_de_type = ['denoising_bsd68_15', 'denoising_bsd68_25', 'denoising_bsd68_50', 'deraining']
elif options.de_type[0] == '3tasks':
    options.de_type = ['denoising_0', 'deraining', 'dehazing']
    options.test_de_type = ['denoising_bsd68_15', 'denoising_bsd68_25', 'denoising_bsd68_50', 'deraining', 'dehazing']
elif options.de_type[0] == '4tasks':
    options.de_type = ['denoising_0', 'deraining', 'dehazing', 'deblurring']
    options.test_de_type = ['denoising_bsd68_15', 'denoising_bsd68_25', 'denoising_bsd68_50', 'deraining', 'dehazing', 'deblurring']

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
        options.lr = 3e-4
elif options.encoder_type in ['Uformer', 'Oformer']:
    
    if options.encoder_dim == None:
        options.encoder_dim = 256
        
    if options.lr == None:
        options.lr = 2e-4
else:
    assert False, "invalid encoder type."
    

if options.frequency_decompose_type.split('_')[0].isdigit() and options.frequency_decompose_type.split('_')[1] == 'bands':
    pass

elif options.frequency_decompose_type == 'DC':
    pass

elif options.frequency_decompose_type == 'none':
    pass

else:
    assert False, "invalid frequency decomposition type."