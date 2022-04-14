from fastapi import FastAPI
from fastapi import UploadFile, File
from predict import read_im, preprocess_im, predict
import uvicorn

app = FastAPI()

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
    prediction = predict(image)
    print(prediction)
    pass

if __name__ == "__main__":
    uvicorn.run(app, port = 8000, host='0.0.0.0')