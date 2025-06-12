import os
from Model.Adapter import *
from Model.Model import *

def set_model(args):
    EEGViT_params = dict(args.Model.EEGViT)
    EEGViT_params['embed_dim'] = args.Train.latent_dim
    EEGViT_params['chunk_size'] = args.Model.SigSize[3]
    EEGViT_params['num_electrodes'] = args.Model.SigSize[2]
    EEGViT_params['width'] = args.Train.latent_dim

    print('EEGViT:',EEGViT_params)

    IMGViT_params = dict(args.Model.IMGViT)
    IMGViT_params['input_resolution'] = args.Model.ImgSize[3]
    IMGViT_params['embed_dim'] = args.Train.latent_dim
    IMGViT_params['width'] = args.Train.latent_dim
    print('IMGViT:',IMGViT_params)

    Adapter_params = dict(args.Model.Adapter)
    Adapter_params['Encoder_option'] = args.Encoder_mode #'both', 'eeg' or 'img'
    Adapter_params['down_size'] = args.Adapter_Size
    Adapter_params['adapter_layernorm_option'] = 'in' #'in' or 'out'
    Adapter_params['adapter_scalar'] = '1.0' #"learnable_scalar" or '1.0'(scaler string)
    Adapter_params['init_option'] = 'lora' #'lora' or 'bert'
    Adapter_params['ffn_option'] = args.Adapter_Mode
    print('Adapter:',Adapter_params)
    
    model = AdapterCLIP(EEGViT_params, IMGViT_params, Adapter_params)
    fname = 'Adapter_256.pt' if args.Encoder_mode == 'both' else 'Adapter_EEG256.pt' 
    PATH = os.path.join('PretrainingWeights', f'sub_{args.TargetSub}', fname)

    model.load_state_dict(torch.load(PATH)['model_state_dict'], strict=False)
            
    model = model.to(torch.device(args.device))
    return model