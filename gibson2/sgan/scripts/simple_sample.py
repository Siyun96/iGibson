import argparse
import os
import torch

from attrdict import AttrDict
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)


from sgan.models import TrajectoryGenerator
from sgan.utils import relative_to_abs, get_dset_path

def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator

def evaluate(args, generator, num_samples):
    with torch.no_grad():
        # batch = [tensor.cuda() for tensor in batch]
        # (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
        #  non_linear_ped, loss_mask, seq_start_end) = batch
        
        cuda0 = torch.device('cuda:0')

        ped_1_obs_traj = torch.tensor([[ 4.3400,  7.0500],
        [ 4.3400,  7.0500],
        [ 4.3400,  7.0500],
        [ 4.3400,  7.0500],
        [ 4.3400,  7.0500],
        [ 4.3400,  7.0500],
        [ 4.3400,  7.0500],
        [ 4.3400,  7.0500]], device=cuda0)

        ped_2_obs_traj = torch.tensor([[ 0.8700,  7.3600],
        [ 1.3300,  7.2700],
        [ 1.7900,  7.2700],
        [ 2.2800,  7.2100],
        [ 2.8100,  7.2100],
        [ 3.2400,  7.1100],
        [ 3.7200,  7.1400],
        [ 4.1400,  7.1400]], device=cuda0)

        #seq start end: [0 peds,2 peds]
        #obs_traj: (obs_len, 2 peds, 2)
        
        obs_traj = torch.stack([ped_1_obs_traj, ped_2_obs_traj], dim = 1)
        obs_traj_rel = obs_traj[1:,:,:] - obs_traj[:-1,:,:]
        seq_start_end = torch.tensor([[0,2]], device = cuda0)

        for _ in range(num_samples):
            pred_traj_fake_rel = generator(
                obs_traj, obs_traj_rel, seq_start_end
            )
            pred_traj_fake = relative_to_abs(
                pred_traj_fake_rel, obs_traj[-1]
            )
            # print("fake_shape")
            # print(pred_traj_fake.shape)

        return obs_traj, pred_traj_fake

# def plot_traj(i, traj):
    

def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        path = get_dset_path(_args.dataset_name, args.dset_type)
        obs_traj, pred_traj_fake = evaluate(_args, generator, args.num_samples)
        print("dim of fake")
        print(pred_traj_fake.shape)
        print("dim of objs")
        print(obs_traj.shape)
        i = 0
        for i in range(0,17):
            plt.figure()
            plt.xlim(0,10)
            plt.ylim(5,9)
            k = i + 4
            for j in range(i, k):
                if j < 8:
                    plt.plot(obs_traj[j][0][0], obs_traj[j][0][1],".",color='blue' )
                    plt.plot(obs_traj[j][1][0], obs_traj[j][1][1],".",color='red')
                else:
                    print(j-8)
                    plt.plot(pred_traj_fake[j-8][0][0], pred_traj_fake[j-8][0][1],"*",color='blue')
                    plt.plot(pred_traj_fake[j-8][1][0], pred_traj_fake[j-8][1][1],"*",color='red')
            plt.savefig(str(i)+".png")
            plt.close()

        print('Dataset: {}, Pred Len: {}'.format(
            _args.dataset_name, _args.pred_len))
        break


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)