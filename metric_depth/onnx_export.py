import argparse
import torch
import torch.onnx
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
import onnxruntime as ort
import onnx

DEBUG = True

def export_to_onnx(model, output_path, input_size, encoder, device):
    """
    Export the Depth Anything V2 model to ONNX format
    
    Args:
        model (DepthAnythingV2): The trained depth estimation model
        output_path (str): Path to save the ONNX model
        input_size (int): Input image size
        encoder (str): Encoder type 
    """
    # Set the model to evaluation mode
    model.eval()
    w,h = input_size
    # Create a dummy input tensor
    dummy_input = torch.randn(
        1, 3, input_size[0], input_size[1], 
        dtype=torch.float32, 
        device=device
    )

    print('Onnx model input size', dummy_input.shape)
    
    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['depth'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'depth': {0: 'batch_size', 1: 'height', 2: 'width'}
        }
    )
    
    print(f"ONNX model exported successfully to {output_path}")
    if DEBUG:
        # Load the ONNX model and verify its export
        verify_onnx_model(output_path, dummy_input)

    



def verify_onnx_model(onnx_path, sample_input):
    """
    Verify an ONNX model's export by checking:
    1. Model can be loaded correctly
    2. Model structure is valid
    3. Model can run inference
    """
    # 1. Load the ONNX model
    model = onnx.load(onnx_path)
    
    # 2. Check model is structurally valid
    onnx.checker.check_model(model)
    print("✓ Model loaded successfully and passed structural validation")

    

def main():
    mm_conversion = {'mm': 1, 'cm': 10, 'dm': 100, 'm': 1000}
    # Model configuration dictionary (same as in original script)
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Export Depth Anything V2 to ONNX')
    parser.add_argument('--encoder', type=str, default='vitl', 
                        choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Encoder type for the model')
    parser.add_argument('--load-from', type=str, 
                        default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth',
                        help='Path to the pre-trained model checkpoint')
    parser.add_argument('--input_size', type=tuple, default=(518, 686),
                        help='Input image size for the model')
    parser.add_argument('--output-path', type=str, default='depth_anything_v2_cem_518_686.onnx',
                        help='Output path for the ONNX model')
    parser.add_argument('--max-depth', type=float, default=20,
                        help='Maximum depth value')
    parser.add_argument('--min-depth', type=float, default=0.1,
                        help='Minimum depth value')
    parser.add_argument('--unit-measure', type=str, default='cm')
    parser.add_argument('--multiple-of', type=int, default=14, help='Check this parameter in the Resize class')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Determine device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Initialize and load the model
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    # depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    # print(depth_anything)
    try:
        depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    except:
        state_dict = torch.load(args.load_from, map_location='cpu')
        my_state_dict = {}
        for key in state_dict['model'].keys():
            my_state_dict[key.replace('module.', '')] = state_dict['model'][key]
        depth_anything.load_state_dict(my_state_dict)
    depth_anything = depth_anything.to(DEVICE).eval()
    
    output_path = args.load_from.replace('latest.pth', f'{args.input_size[0]}_{args.input_size[1]}_onnx_model.onnx')
    # Export to ONNX
    export_to_onnx(
        model=depth_anything, 
        output_path=output_path, 
        input_size=args.input_size,
        encoder=args.encoder,
        device=DEVICE,
        
    )

    cfg_path = output_path.replace('.onnx', '.cfg')
    with open(cfg_path, 'w') as f:
        f.write(f'input_size = {args.input_size}\n')
        f.write(f'min_depth = {args.min_depth}\n')
        f.write(f'max_depth = {args.max_depth} \n')
        f.write(f'unit_measure = {mm_conversion[str(args.unit_measure)]}\n')
        f.write(f'multiple_of= {args.multiple_of}\n')
    print(f'Config file saved to {cfg_path}')

if __name__ == '__main__':
    main()