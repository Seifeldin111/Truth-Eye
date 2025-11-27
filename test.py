from insightface.app import FaceAnalysis
import onnx

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=-1)

print(app.models.keys())
