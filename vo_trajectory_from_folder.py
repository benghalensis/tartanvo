from torch.utils.data import DataLoader
from Datasets.utils import ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow, plot_traj, visflow
from Datasets.tartanTrajFlowDataset import TrajFolderDataset
from Datasets.transformation import ses2poses_quat
from evaluator.tartanair_evaluator import TartanAirEvaluator
from TartanVO import TartanVO

import argparse
import numpy as np
import cv2
from os import mkdir, makedirs
from os.path import isdir, join
import wandb
import shutil

def get_args():
    parser = argparse.ArgumentParser(description='HRL')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size (default: 1)')
    parser.add_argument('--worker-num', type=int, default=1,
                        help='data loader worker number (default: 1)')
    parser.add_argument('--image-width', type=int, default=640,
                        help='image width (default: 640)')
    parser.add_argument('--image-height', type=int, default=448,
                        help='image height (default: 448)')
    parser.add_argument('--model-name', default='',
                        help='name of pretrained model (default: "")')
    parser.add_argument('--euroc', action='store_true', default=False,
                        help='euroc test (default: False)')
    parser.add_argument('--kitti', action='store_true', default=False,
                        help='kitti test (default: False)')
    parser.add_argument('--kitti-intrinsics-file',  default='',
                        help='kitti intrinsics file calib.txt (default: )')
    parser.add_argument('--test-dir', default='',
                        help='test trajectory folder where the RGB images are (default: "")')
    parser.add_argument('--flow-dir', default='',
                        help='test trajectory folder where the RGB images are (default: "")')
    parser.add_argument('--pose-file', default='',
                        help='test trajectory gt pose file, used for scale calculation, and visualization (default: "")')
    parser.add_argument('--save-flow', action='store_true', default=False,
                        help='save optical flow (default: False)')
    parser.add_argument('--save-path', type=str, default="",
                        help='save optical flow (default: False)')
    parser.add_argument('--skip-n', type=int, default=0,
                        help='If the value is 0, it skips no frames, if 1 then it skips every other frame, if 2 then it skips 2 and processes 1')

    args = parser.parse_args()

    return args



def get_l1_loss(pred, gt):
    # Scale the pred where the biggest dimension matches the biggest dimension of the gt
    if gt.shape[0] > gt.shape[1]:
        scale_factor = gt.shape[0] / pred.shape[0]
    else:
        scale_factor = gt.shape[1] / pred.shape[1]

    if scale_factor != 1:
        new_width = int(pred.shape[1] * scale_factor)
        new_height = int(pred.shape[0] * scale_factor)
        new_dimension = (new_width, new_height)

        cv2.resize(pred, new_dimension, interpolation=cv2.INTER_LINEAR)

    return np.mean(np.abs(pred-gt))

"""
Example to run the script:
python vo_trajectory_from_folder.py --model-name tartanvo_1914.pkl --batch-size 1 --worker-num 1 
--test-dir /ocean/projects/cis220039p/shared/tartanair_v2_event/CountryHouseAutoExposure/Data_easy/P000/events/reconstruction 
--flow-dir /ocean/projects/cis220039p/shared/tartanair_v2_event/CountryHouseAutoExposure/Data_easy/P000/flow_lcam_front 
--skip-n 1 --save-flow
"""
if __name__ == '__main__':
    args = get_args()

    testvo = TartanVO(args.model_name)

    # load trajectory data from a folder
    datastr = 'tartanair'
    if args.kitti:
        datastr = 'kitti'
    elif args.euroc:
        datastr = 'euroc'
    else:
        datastr = 'tartanair'
    focalx, focaly, centerx, centery = dataset_intrinsics(datastr) 
    if args.kitti_intrinsics_file.endswith('.txt') and datastr=='kitti':
        focalx, focaly, centerx, centery = load_kiiti_intrinsics(args.kitti_intrinsics_file)

    transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])

    testDataset = TrajFolderDataset(args.test_dir, flow_folder=args.flow_dir, posefile = args.pose_file, transform=transform, 
                                        focalx=focalx, focaly=focaly, centerx=centerx, centery=centery, skip_n=args.skip_n)
    testDataloader = DataLoader(testDataset, batch_size=args.batch_size, 
                                        shuffle=False, num_workers=args.worker_num)
    testDataiter = iter(testDataloader)

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="tartanvo",
        
        # track hyperparameters and run metadata
        config={
        "test-dir": args.test_dir,
        }
    )

    motionlist = []
    testname = datastr + '_' + args.model_name.split('.')[0]
    if args.save_flow:
        if args.save_path == "":
            flowdir = join(args.test_dir, "flow")
        else:
            flowdir = args.save_path
        # If flowdir exists, delete it and make a new one
        if isdir(flowdir):
            shutil.rmtree(flowdir)
        makedirs(flowdir)
        flowcount = 0
    
    while True:
        try:
            sample = next(testDataiter)
        except StopIteration:
            break

        motions, flow = testvo.test_batch(sample)
        motionlist.extend(motions)


        if (args.save_flow) and ('flow' in sample):
            for k in range(flow.shape[0]):
                flowk = flow[k].transpose(1,2,0)
                gtflowk = sample['flow'][k].numpy().transpose(1,2,0)

                # Calculate loss and log it
                val_loss = get_l1_loss(flowk, gtflowk)
                wandb.log({"val_loss": val_loss})
                
                # np.save(flowdir+'/'+str(flowcount).zfill(6)+'.npy',flowk)
                flowk_vis = visflow(flowk)
                gtflowk_vis = visflow(gtflowk)

                cv2.imwrite(flowdir+'/'+str(flowcount).zfill(6)+'.png', np.hstack((flowk_vis, gtflowk_vis)))
                flowcount += 1
    
    # Close wandb
    wandb.finish()
    poselist = ses2poses_quat(np.array(motionlist))

    # calculate ATE, RPE, KITTI-RPE
    if args.pose_file.endswith('.txt'):
        evaluator = TartanAirEvaluator()
        results = evaluator.evaluate_one_trajectory(args.pose_file, poselist, scale=True, kittitype=(datastr=='kitti'))
        if datastr=='euroc':
            print("==> ATE: %.4f" %(results['ate_score']))
        else:
            print("==> ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" %(results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]))

        # save results and visualization
        plot_traj(results['gt_aligned'], results['est_aligned'], vis=False, savefigname='results/'+testname+'.png', title='ATE %.4f' %(results['ate_score']))
        np.savetxt('results/'+testname+'.txt',results['est_aligned'])
    else:
        np.savetxt('results/'+testname+'.txt',poselist)
