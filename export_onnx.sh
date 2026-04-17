python onnx_export.py --encoder vitb --load-from /home/elena/repos/Rim-Split/outputs/20260320_164637_conti4porte_vitb_depth/latest.pth --output-path vitb_ferrari.onnx --max-depth 1.0 --min-depth 0.01
pip index onnxsim
onnxsim original_model.onnx simplified_model.onnx