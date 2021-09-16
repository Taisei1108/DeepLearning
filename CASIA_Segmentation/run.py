import argparse
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #Common args
    parser.add_argument('--dataset_root', required=True,type=str,
                        help='input Dataset path for training') 
    parser.add_argument('--cuda_port', type=str, choices=['cuda:0','cuda:1'], required=True,
                    help='choose cuda port for  CUDA training')
    #train_cam
    parser.add_argument('--cam_batch-size', type=int, default=16)
    parser.add_argument('--cam_network', type=str, default='resnet101')
    parser.add_argument('--cam_crop_size', type=int, default=256)
    parser.add_argument('--cam_output_class', type=int, default=2)
    parser.add_argument('--cam_learning_rate', type=float, default=1e-5)  
    parser.add_argument('--cam_momentum', type=int, default=0.99, metavar='M')
    parser.add_argument('--cam_num_epochs', type=int, default=150)                                                                                     
    parser.add_argument('--cam_affine_degree', type=int, default=10)     
    parser.add_argument('--cam_scale', default=(1.0, 0.5, 1.5, 2.0),
                    help='マルチスケールに対応')       
    
    # Output Path
    parser.add_argument("--cam_weights_name", default="sess/res50_cam.pth", type=str)
    parser.add_argument("--cam_out_dir", default="result/cam", type=str)
    
    # Step
    parser.add_argument("--train_cam_pass", default=True)

    args = parser.parse_args()

    os.makedirs("sess", exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)

    print(vars(args))
    if args.train_cam_pass is True:
        import train_cam
        train_cam.run(args)