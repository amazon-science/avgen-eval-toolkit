from omegaconf import OmegaConf
from tqdm import tqdm
import os
import numpy as np
import shutil

from utils.utils import build_cfg_path, form_list_from_user_input, sanity_check


def main(args_cli):

    args_yml = OmegaConf.load(build_cfg_path('i3d'))
    args = OmegaConf.merge(args_yml, args_cli)  # the latter arguments are prioritized
    from models.i3d.extract_i3d import ExtractI3D as Extractor

    extractor = Extractor(args)
    video_paths = []


    for root, dirs, files in os.walk(args.video_paths):
        for file in files:
            video_paths.append(os.path.abspath(os.path.join(root, file)))
    

    print(f'The number of specified videos: {len(video_paths)}')

    for video_path in tqdm(video_paths):
        extractor._extract(video_path)  # note the `_` in the method name


    # Directory containing the files
    input_directory = './output_feats/i3d'
    output_directory = './output_feats/i3d_feats'

    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # List files that end with 'rgb.npy'
    rgb_files = [f for f in os.listdir(input_directory) if f.endswith('_rgb.npy')]

    for rgb_file in rgb_files:
        # Construct flow file name based on rgb file name
        base_name = rgb_file.rsplit('_', 1)[0]
        flow_file = base_name + '_flow.npy'

        # Make sure the flow file exists
        if flow_file in os.listdir(input_directory):
            # Load the feature vectors
            rgb_data = np.load(os.path.join(input_directory, rgb_file))
            flow_data = np.load(os.path.join(input_directory, flow_file))

            # Check if their shapes match, if not continue to next pair
            if rgb_data.shape != flow_data.shape:
                print(f"Shapes of {rgb_file} and {flow_file} do not match. Skipping.")
                continue

            # Add the feature vectors
            combined_data = rgb_data + flow_data

            # Save the combined data
            output_file = os.path.join(output_directory, base_name + '_i3d.npy')
            np.save(output_file, combined_data)
    
    shutil.rmtree(input_directory)

    # args.feature_type = 'vggish'
    args_yml = OmegaConf.load(build_cfg_path('vggish'))
    args = OmegaConf.merge(args_yml, args_cli)  # the latter arguments are prioritized
    from models.vggish.extract_vggish import ExtractVGGish as Extractor
    
    extractor = Extractor(args)

    for video_path in tqdm(video_paths):
        extractor._extract(video_path)  # note the `_` in the method name
    # yep, it is this simple!


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    args_cli = OmegaConf.from_cli()
    main(args_cli)



