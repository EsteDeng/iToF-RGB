import argparse, random, os

def load_args():
    parser = argparse.ArgumentParser(description='Split dataset into train and val')
    parser.add_argument('--data_root', type=str, default='ToFFlyingThings3D/',
                        help='Directory path to ToFFlyingThings3D')
    parser.add_argument('--split', type=float, default=0.95, help='train-val split ratio')
    args = parser.parse_args()
    return args

def split_dataset(args):
    list_path = os.path.join(args.data_root, 'train_list.txt')
    with open(list_path, 'r') as f:
        filenames = f.read().split('\n')
        filenames = [f for f in filenames if f != '']
    backup_list_path = os.path.join(args.data_root, 'backup_train_list.txt')
    os.rename(list_path, backup_list_path)

    random.shuffle(filenames)
    split_idx = int(len(filenames) * args.split)
    train_filenames = filenames[:split_idx]
    val_filenames = filenames[split_idx:]

    with open(os.path.join(args.data_root, 'train_list.txt'), 'w') as f:
        f.write('\n'.join(train_filenames))

    with open(os.path.join(args.data_root, 'val_list.txt'), 'w') as f:
        f.write('\n'.join(val_filenames))

def main():
    args = load_args()
    split_dataset(args)

if __name__ == '__main__':
    main()