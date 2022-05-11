from fastapi import FastAPI
from fastapi import UploadFile, File
from predict import read_im, preprocess_im, predict
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
import os

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers    
)

@app.get('/')
def greet():
    return f"Welcome to phytomonk classification server"

@app.post('/api/predict')
async def predict_disease(file: UploadFile = File(...)):
    #read the uploaded image
    image = read_im(await file.read())
    #do some preprocessing
    image = preprocess_im(image)
    #do prediction
    im_class, pred = predict(image)
    #class name for prediction
    classes = ["bacterial_spot", "healthy", "mildew", "rust"]
    resp = {}
    resp["index"] = f"{im_class}"
    resp["class"] = classes[im_class]
    resp["confidence"] = f"{pred[0][im_class]*100:0.2f} %"
    
    return resp

if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	run(app, host="0.0.0.0", port=port)