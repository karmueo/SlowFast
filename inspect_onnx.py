import onnx
import sys

model_path = "output/custom_x3d_3/model_simplified.onnx"
try:
    model = onnx.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

print(f"Model: {model_path}")
print(f"IR Version: {model.ir_version}")
print(f"Opset: {model.opset_import[0].version}")

print("\nLast 10 nodes:")
for i, node in enumerate(model.graph.node[-10:]):
    print(f"Node {i}: {node.op_type} ({node.name})")
    for attr in node.attribute:
        if attr.name in ['axis', 'axes', 'keepdims']:
            if attr.type == onnx.AttributeProto.INT:
                print(f"  {attr.name}: {attr.i}")
            elif attr.type == onnx.AttributeProto.INTS:
                print(f"  {attr.name}: {attr.ints}")

print("\nOutput info:")
for output in model.graph.output:
    print(f"  Name: {output.name}")
    print(f"  Type: {output.type}")
