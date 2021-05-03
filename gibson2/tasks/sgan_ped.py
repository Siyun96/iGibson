# from gibson2.episodes.episode_sample import SocialNavEpisodesConfig
# from gibson2.tasks.point_nav_random_task import PointNavRandomTask
# from gibson2.objects.visual_marker import VisualMarker
# from gibson2.objects.pedestrian import Pedestrian
# from gibson2.termination_conditions.pedestrian_collision import PedestrianCollision
# from gibson2.utils.utils import l2_distance
from gibson2.sgan.sgan.models import TrajectoryGenerator
from gibson2.sgan.sgan.utils import relative_to_abs, get_dset_path

import torch
from collections import defaultdict
import numpy as np
import argparse
import os


# historical traj
# output samples
# TODO: concact goal to sgan
def gen_ped_data(generator, ped_dict, num_samples, next_goal = [(0, 0)]):
    data = []
    cuda0 = torch.device('cuda:0')
    for ped in ped_dict.keys():
        data.append(torch.tensor(ped_dict[ped], device = cuda0))
    obs_traj = torch.stack(data, dim = 1)
    obs_traj_rel = obs_traj[1:,:,:] - obs_traj[:-1,:,:]
    seq_start_end = torch.tensor([[0,len(ped_dict)]], device = cuda0)
    #pred_len = 1
    mod_ped_pos = []
    for _ in range(num_samples):
        pred_traj_fake_rel = generator(
            obs_traj, obs_traj_rel, seq_start_end
        )
        pred_traj_fake = relative_to_abs(
            pred_traj_fake_rel, obs_traj[-1]
        )
        # sample_id, ped_id, position
        mod_ped_pos.append(pred_traj_fake.detach().cpu().numpy())

    return mod_ped_pos


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--num_samples', default=20, type=int)
    parser.add_argument('--dset_type', default='test', type=str)
    args = parser.parse_args()
    main(args)