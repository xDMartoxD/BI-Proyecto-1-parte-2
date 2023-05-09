from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI()

from model import p1
import pandas as pd



@app.get("/review")
def review(review: str) :
   return {"sentiment": p1.predict([review])[0] }


@app.post("/upload-csv/")
async def read_csv(file: UploadFile | None = None):
    if not file:
         return {"message": "No upload file sent"}
    else:
         file_path = "C:/Users/marti/Desktop/projects/bi_p1_parte2/data/MovieReviews.csv"

    # Crear un objeto de respuesta de archivo con el contenido del archivo CSV
         response = FileResponse(file_path, filename="response.csv")

    # Establecer las cabeceras de la respuesta para indicar que es un archivo CSV
         response.headers["Content-Disposition"] = "attachment; filename=response.csv"
         response.headers["Content-Type"] = "text/csv"

    # Retornar el objeto de respuesta de archivo
         return response
    