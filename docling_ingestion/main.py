import os
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

app = FastAPI(title="Docling Microservice")

# Initialize converter once on startup with OCR disabled
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False
pipeline_options.do_table_structure = True

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

@app.post("/parse")
async def parse_pdf(file: UploadFile = File(...)):
    try:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        # Convert using Docling
        result = converter.convert(temp_path)
        markdown_text = result.document.export_to_markdown()

        # Clean up
        os.remove(temp_path)

        return {"filename": file.filename, "markdown": markdown_text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
def health_check():
    return {"status": "ok"}
