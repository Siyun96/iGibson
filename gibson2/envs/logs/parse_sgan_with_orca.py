import os
import re
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_name', type=str, default='sgan_with_orca')
    parser.add_argument('--filename', type=str, default='ped_trajectories.txt')
    parser.add_argument('--num_ped', type=int, default=2)
    args = parser.parse_args()

    step_start = re.compile('.* Entering Step: Ped ID: (\d), location: \((.*), (.*)\)')
    # ORCA
    use_orca = re.compile('.* Default to use ORCA to set velocity')
    orca_desired_vel = re.compile('.* ORCA desired velocity for ped (\d): \((.*), (.*)\)')
    orca_move_accepted = re.compile('.* Next move accepted for ped (\d) at location \[(.*)\]')
    orca_move_rejected = re.compile('.* Next move rejected for ped (\d)')
    # SGAN
    use_sgan = re.compile('.* Use SGAN to generate preferred velocity')
    sgan_sample_desired_vel = re.compile('.* SGAN sample idx (\d) desired velocity for ped (\d): \[(.*)\]')
    sgan_sample_success = re.compile('.* SGAN sample idx (\d) success')
    sgan_sample_failed = re.compile('.* Next move rejected for ped (\d)')
    sgan_no_valid_sample = re.compile('.* SGAN failed to generate acceptable samples. Using ORCA to set velocity')
    sgan_set_velocity = re.compile('.* Using SGAN to set velocity')
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
            if counter >= 500:
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
                    sample_success = False
                    no_valid_sample = False
                    desired_vel_temp = {}
                    for i in range(args.num_ped):
                        desired_vel_temp[i] = []
                    while (not sample_success) and (not no_valid_sample):
                        for i in range(args.num_ped):
                            m = sgan_sample_desired_vel.match(line)
                            sample_idx, pid, xy = m.groups()
                            x, y = xy.strip().split()
                            desired_vel_temp[int(pid)].append((f'sgan_{sample_idx}', float(x), float(y)))
                            line = f.readline()
                        if sgan_sample_success.match(line):
                            sample_success = True
                            for i in range(args.num_ped):
                                desired_velocity[i].append(desired_vel_temp[i][-1])
                        if sgan_sample_failed.match(line):
                            line = f.readline()
                            if sgan_no_valid_sample.match(line):
                                no_valid_sample = True
                            else:
                                line = f.readline()
                    if sample_success:
                        line = f.readline()
                        assert(sgan_set_velocity.match(line))
                        line = f.readline()
                        for i in range(args.num_ped):
                            m = sgan_set_next_step.match(line)
                            pid, xyz = m.groups()
                            x, y, _ = xyz.strip().split()
                            move_success[int(pid)].append(('success', float(x), float(y)))
                            line = f.readline()
                    elif no_valid_sample:
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
                    else:
                        print('Unexpected SGAN sample read')
                        assert(0)
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