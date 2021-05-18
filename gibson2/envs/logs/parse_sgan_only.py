import os
import re
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_name', type=str, default='sgan')
    parser.add_argument('--filename', type=str, default='ped_trajectories.txt')
    parser.add_argument('--num_ped', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=100)
    args = parser.parse_args()

    step_start = re.compile('.* Entering Step: Ped ID: (\d), location: \((.*), (.*)\)')
    # ORCA
    use_orca = re.compile('.* Default to use ORCA to set velocity')
    orca_desired_vel = re.compile('.* ORCA desired velocity for ped (\d): \((.*), (.*)\)')
    orca_move_accepted = re.compile('.* Next move accepted for ped (\d) at location \[(.*)\]')
    orca_move_rejected = re.compile('.* Next move rejected for ped (\d)')
    # SGAN
    use_sgan = re.compile('.* Use SGAN to generate next location')
    sgan_sample_next_pos = re.compile('.* SGAN sample idx (\d) next position for ped (\d): \[(.*)\]')
    sgan_set_velocity = re.compile('.* Using SGAN to set next location')
    sgan_set_next_step = re.compile('.* Set next step ped (\d) at location \[(.*)\]')

    ped_pos_dict = {}
    desired_velocity = {}
    move_success = {}
    for i in range(args.num_ped):
        ped_pos_dict[i] = []
        desired_velocity[i] = []
        move_success[i] = []

    counter = 0
    with open(os.path.join(args.dir_name, args.filename), 'r') as f:
        line = f.readline()
        while line:
            if counter >= args.num_steps:
                break
            if step_start.match(line):
                counter += 1
                for i in range(args.num_ped):
                    m = step_start.match(line)
                    pid, x, y = m.groups()      # all of type str
                    ped_pos_dict[int(pid)].append((float(x), float(y)))
                    line = f.readline()
                if use_orca.match(line):
                    line = f.readline()
                    for i in range(args.num_ped):
                        m = orca_desired_vel.match(line)
                        pid, x, y = m.groups()
                        desired_velocity[int(pid)].append(('orca', float(x), float(y)))
                        line = f.readline()
                    for i in range(args.num_ped):
                        m = orca_move_accepted.match(line)
                        if m:
                            pid, xyz = m.groups()
                            x, y, _ = xyz.strip().split()
                            move_success[int(pid)].append(('success', float(x), float(y)))
                        else:
                            m = orca_move_rejected.match(line)
                            pid = m.group(1)
                            move_success[int(pid)].append(('fail'))
                        line = f.readline()
                elif use_sgan.match(line):
                    line = f.readline()
                    for i in range(args.num_ped):
                        m = sgan_sample_next_pos.match(line)
                        sample_idx, pid, xyz = m.groups()
                        x, y, _ = xyz.strip().split()
                        desired_velocity[int(pid)].append((f'sgan_{sample_idx}', float(x), float(y)))
                        line = f.readline()
                    assert(sgan_set_velocity.match(line))
                    line = f.readline()
                    for i in range(args.num_ped):
                        m = sgan_set_next_step.match(line)
                        pid, xyz = m.groups()
                        x, y, _ = xyz.strip().split()
                        move_success[int(pid)].append(('success', float(x), float(y)))
                        line = f.readline()
                else:
                    print(line)
                    print('Unexpected match')
            
            else:
                line = f.readline()

    assert(counter == len(ped_pos_dict[0]))
    assert(counter == len(desired_velocity[0]))
    assert(counter == len(move_success[0]))

    for i in range(args.num_ped):
        print(f'Pedestrian {i}')
        for t in range(len(ped_pos_dict[i])):
            print(ped_pos_dict[i][t], desired_velocity[i][t], move_success[i][t])
        print('\n\n')