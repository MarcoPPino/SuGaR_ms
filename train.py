import argparse
import os
import time
from sugar_utils.general_utils import str2bool
from sugar_trainers.coarse_density import coarse_training_with_density_regularization
from sugar_trainers.coarse_sdf import coarse_training_with_sdf_regularization
from sugar_extractors.coarse_mesh import extract_mesh_from_coarse_sugar
from sugar_trainers.refine import refined_training
from sugar_extractors.refined_mesh import extract_mesh_and_texture_from_refined_sugar


class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self


if __name__ == "__main__":
    # ----- Parser -----
    parser = argparse.ArgumentParser(description='Script to optimize a full SuGaR model.')
    
    # Data and vanilla 3DGS checkpoint
    parser.add_argument('-s', '--scene_path',
                        type=str, 
                        help='(Required) path to the scene data to use.')  
    parser.add_argument('-c', '--checkpoint_path',
                        type=str, 
                        help='(Required) path to the vanilla 3D Gaussian Splatting Checkpoint to load.')
    parser.add_argument('-i', '--iteration_to_load', 
                        type=int, default=7000, 
                        help='iteration to load.')
    parser.add_argument('-o', '--output_dir',
                        type=str, 
                        default=None,
                        help='OutputPath brudi')
    
    # Regularization for coarse SuGaR
    parser.add_argument('-r', '--regularization_type', type=str,
                        help='(Required) Type of regularization to use for coarse SuGaR. Can be "sdf" or "density". ' 
                        'For reconstructing detailed objects centered in the scene with 360° coverage, "density" provides a better foreground mesh. '
                        'For a stronger regularization and a better balance between foreground and background, choose "sdf".')
    
    # Extract mesh
    parser.add_argument('-l', '--surface_level', type=float, default=0.3, 
                        help='Surface level to extract the mesh at. Default is 0.3')
    parser.add_argument('-v', '--n_vertices_in_mesh', type=int, default=1_000_000, 
                        help='Number of vertices in the extracted mesh.')
    parser.add_argument('-b', '--bboxmin', type=str, default=None, 
                        help='Min coordinates to use for foreground.')  
    parser.add_argument('-B', '--bboxmax', type=str, default=None, 
                        help='Max coordinates to use for foreground.')
    parser.add_argument('--center_bbox', type=str2bool, default=True, 
                        help='If True, center the bbox. Default is False.')
    
    # Parameters for refined SuGaR
    parser.add_argument('-g', '--gaussians_per_triangle', type=int, default=1, 
                        help='Number of gaussians per triangle.')
    parser.add_argument('-f', '--refinement_iterations', type=int, default=15_000, 
                        help='Number of refinement iterations.')
    
    # (Optional) Parameters for textured mesh extraction
    parser.add_argument('-t', '--export_uv_textured_mesh', type=str2bool, default=True, 
                        help='If True, will export a textured mesh as an .obj file from the refined SuGaR model. '
                        'Computing a traditional colored UV texture should take less than 10 minutes.')
    parser.add_argument('--square_size',
                        default=10, type=int, help='Size of the square to use for the UV texture.')
    parser.add_argument('--postprocess_mesh', type=str2bool, default=False, 
                        help='If True, postprocess the mesh by removing border triangles with low-density. '
                        'This step takes a few minutes and is not needed in general, as it can also be risky. '
                        'However, it increases the quality of the mesh in some cases, especially when an object is visible only from one side.')
    parser.add_argument('--postprocess_density_threshold', type=float, default=0.1,
                        help='Threshold to use for postprocessing the mesh.')
    parser.add_argument('--postprocess_iterations', type=int, default=5,
                        help='Number of iterations to use for postprocessing the mesh.')
    
    # (Optional) PLY file export
    parser.add_argument('--export_ply', type=str2bool, default=True,
                        help='If True, export a ply file with the refined 3D Gaussians at the end of the training. '
                        'This file can be large (+/- 500MB), but is needed for using the dedicated viewer. Default is True.')
    
    # (Optional) Default configurations
    parser.add_argument('--low_poly', type=str2bool, default=False, 
                        help='Use standard config for a low poly mesh, with 200k vertices and 6 Gaussians per triangle.')
    parser.add_argument('--mid_poly', type=str2bool, default=False,
                        help='Use standard config for a high poly mesh, with 1M vertices and 1 Gaussians per triangle.')
    parser.add_argument('--high_poly', type=str2bool, default=False,
                        help='Use standard config for a high poly mesh, with 1M vertices and 1 Gaussians per triangle.')
    parser.add_argument('--ultra_poly', type=str2bool, default=False,
                        help='Use standard config for a high poly mesh, with 1M vertices and 1 Gaussians per triangle.')
    parser.add_argument('--refinement_time', type=str, default=None, 
                        help="Default configs for time to spend on refinement. Can be 'short', 'medium' or 'long'.")

    parser.add_argument('--zzz', type=str2bool, default=False, 
                        help='puts PC into hibernation after finishing')
      
    # Evaluation split
    parser.add_argument('--eval', type=str2bool, default=True, help='Use eval split.')

    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')
    parser.add_argument('--white_background', type=str2bool, default=False, help='Use a white background instead of black.')

    # Parse arguments
    args = parser.parse_args()
    if args.low_poly:
        args.n_vertices_in_mesh = 200_000
        args.gaussians_per_triangle = 6
        print('Using low poly config.')
    if args.mid_poly:
        args.n_vertices_in_mesh = 700_000
        args.gaussians_per_triangle = 3
        print('Using mid poly config.')
    if args.high_poly:
        args.n_vertices_in_mesh = 1_000_000
        args.gaussians_per_triangle = 1
        print('Using high poly config.')
    if args.ultra_poly:
        args.n_vertices_in_mesh = 3_000_000
        args.gaussians_per_triangle = 1
        print('Using ultra poly config.')


    if args.refinement_time == 'test':
        args.refinement_iterations = 200
        print('Using short refinement time.')
    if args.refinement_time == 'short':
        args.refinement_iterations = 2_000
        print('Using short refinement time.')
    if args.refinement_time == 'medium':
        args.refinement_iterations = 7_000
        print('Using medium refinement time.')
    if args.refinement_time == 'long':
        args.refinement_iterations = 15_000
        print('Using long refinement time.')
    if args.export_uv_textured_mesh:
        print('Will export a UV-textured mesh as an .obj file.')
    if args.export_ply:
        print('Will export a ply file with the refined 3D Gaussians at the end of the training.')




    # ----- Optimize coarse SuGaR -----
    coarse_args = AttrDict({
        'checkpoint_path': args.checkpoint_path,
        'scene_path': args.scene_path,
        'iteration_to_load': args.iteration_to_load,
        'output_dir': args.output_dir,
        'eval': args.eval,
        'estimation_factor': 0.2,
        'normal_factor': 0.2,
        'gpu': args.gpu,
        'white_background': args.white_background,
    })

    #create path for first directory TODO: check 
    if args.output_dir is None:
        if len(args.scene_path.split(os.sep)[-1]) > 0:
            outputStarter = os.path.join("./output/coarse", args.scene_path.split(os.sep)[-1])
        else:
            outputStarter = os.path.join("./output/coarse", args.scene_path.split(os.sep)[-2])
    else:
        outputStarter = args.output_dir


    if args.regularization_type == 'sdf':        
        print("checking coarse_training_sdf...")
        #check if files already exist
        dirName = f'sugarcoarse_3Dgs{args.iteration_to_load}_sdfestimXX_sdfnormYY/'
        dirName = os.path.join(outputStarter, dirName)
        dirName = dirName.replace(
            'XX', str(0.2).replace('.', '')
            ).replace(
                'YY', str(0.2).replace('.', '')
                )
        fp = os.path.join(dirName, '15000.pt')

        if os.path.isfile(fp):
            #trained already
            print("SDF 15000pt already exists, skipping training...")
            coarse_sugar_path = fp
        else:
            #not trained already
            print("nothing found, STARTING SDF coarse training...")
            coarse_sugar_path = coarse_training_with_sdf_regularization(coarse_args)
    
        
    elif args.regularization_type == 'density':
        print("checking coarse_training_density...")
        #check if files already exist
        dirName = f'sugarcoarse_3Dgs{args.iteration_to_load}_densityestimXX_sdfnormYY/'
        dirName = os.path.join(outputStarter, dirName)
        dirName = dirName.replace(
            'XX', str(0.2).replace('.', '')
            ).replace(
                'YY', str(0.2).replace('.', '')
                )
        fp = os.path.join(dirName, '15000.pt')
        if os.path.isfile(fp):
            #trained already
            print("DENSE 15000pt already exists, skipping training...")
            coarse_sugar_path = fp
        else:
            #not trained already
            print("nothing found, STARTING density coarse training...")
            coarse_sugar_path = coarse_training_with_density_regularization(coarse_args)

    else:
        raise ValueError(f'Unknown regularization type: {args.regularization_type}')
    
    
    # ----- Extract mesh from coarse SuGaR -----
    coarse_mesh_args = AttrDict({
        'scene_path': args.scene_path,
        'checkpoint_path': args.checkpoint_path,
        'iteration_to_load': args.iteration_to_load,
        'coarse_model_path': coarse_sugar_path,
        'surface_level': args.surface_level,
        'decimation_target': args.n_vertices_in_mesh,
        'mesh_output_dir': args.output_dir,
        'bboxmin': args.bboxmin,
        'bboxmax': args.bboxmax,
        'center_bbox': args.center_bbox,
        'gpu': args.gpu,
        'eval': args.eval,
        'use_centers_to_extract_mesh': False,
        'use_marching_cubes': False,
        'use_vanilla_3dgs': False,
    })
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("STARTING extract_mesh_from_coarse_sugar")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


    # #check if file has been generated before
    # print(f'files in {args.output_dir} :')
    # print(os.listdir(args.output_dir))
    # foundPLY = False
    # for filename in os.listdir(args.output_dir):
    #     if filename.endswith('.ply'):
    #         #found file, proceeding without extraction
    #         print(f'Found file proceeding to refine with: {os.path.join(args.output_dir, filename)}')
    #         foundPLY = True

    # if foundPLY:
    #     coarse_mesh_path = os.path.join(args.output_dir, filename)
    # else:
    #     print("could not find anything proceeding to extract mesh")
    #     coarse_mesh_path = extract_mesh_from_coarse_sugar(coarse_mesh_args)[0]


    coarse_mesh_path = extract_mesh_from_coarse_sugar(coarse_mesh_args)[0]
    
    
    # ----- Refine SuGaR -----
    refined_args = AttrDict({
        'scene_path': args.scene_path,
        'checkpoint_path': args.checkpoint_path,
        'mesh_path': coarse_mesh_path,      
        'output_dir': args.output_dir,
        'iteration_to_load': args.iteration_to_load,
        'normal_consistency_factor': 0.1,    
        'gaussians_per_triangle': args.gaussians_per_triangle,        
        'n_vertices_in_fg': args.n_vertices_in_mesh,
        'refinement_iterations': args.refinement_iterations,
        'bboxmin': args.bboxmin,
        'bboxmax': args.bboxmax,
        'export_ply': args.export_ply,
        'eval': args.eval,
        'gpu': args.gpu,
        'white_background': args.white_background,
    })
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("STARTING refined_training")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    refined_sugar_path = refined_training(refined_args)
    
    
    # ----- Extract mesh and texture from refined SuGaR -----
    if args.export_uv_textured_mesh:
        refined_mesh_args = AttrDict({
            'scene_path': args.scene_path,
            'iteration_to_load': args.iteration_to_load,
            'checkpoint_path': args.checkpoint_path,
            'refined_model_path': refined_sugar_path,
            'coarse_model_path': coarse_sugar_path,
            'mesh_output_dir': args.output_dir,
            'n_gaussians_per_surface_triangle': args.gaussians_per_triangle,
            'square_size': args.square_size,
            'eval': args.eval,
            'gpu': args.gpu,
            'postprocess_mesh': args.postprocess_mesh,
            'postprocess_density_threshold': args.postprocess_density_threshold,
            'postprocess_iterations': args.postprocess_iterations,
        })
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("STARTING extract_mesh_and_texture_from_refined_sugar")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        refined_mesh_path = extract_mesh_and_texture_from_refined_sugar(refined_mesh_args)


    #put pc to sleep after execution
    if args.zzz:
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("All done Boss! Going to hibernation in 30 seconds")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        time.sleep(30)
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
    
        