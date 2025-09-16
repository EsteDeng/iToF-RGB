import argparse, yaml
import torch
import data, models, tester
from torchvision import transforms
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

def load_args():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--data_root', type=str, default='ToFFlyingThings3D',
                        help='Directory path to ToFFlyingThings3D')
    parser.add_argument('--weight_path', type=str)
    parser.add_argument('--config_path', type=str,
                        help='path to configuration yaml file')
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--output', type=str, default='output_metric.txt',
                        help='output file name')

    parser.add_argument('--n_threads', type=int, default= 8)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = load_args()
    with open(args.config_path) as f:
        conf = yaml.safe_load(f)

    print('init models')
    if conf['mode'] == 'RGBD':
        test_model = models.model.RGBD_Model(**conf['model'])
        test_model.load_state_dict(torch.load(args.weight_path))
        test_model.to(device)
        
    print('init tester')
    Tester = tester.SingleModelTester_tft(conf['data']['cam_path'], 
                                        conf['data']['depth_scale'], 
                                        test_model,
                                        mode=conf['mode']
                                    ).to(device)

    print('init dataset')
    val_dataset  = data.ToF_FlyingTings3D_UW_Simple_Dataset(args.data_root, 'test')
    val_dataset  = torch.utils.data.DataLoader(val_dataset, 
                                            batch_size=args.test_batch_size, 
                                            shuffle=False, 
                                            num_workers=args.n_threads, 
                                            drop_last=False)
    print('start testing')
    metrics = Tester.test(val_dataset)
    for key, value in metrics.items():
        print(key, value)
    with open(args.output, 'w') as f:
        for key, value in metrics.items():
            f.write(key + ': ' + str(value) + '\n')