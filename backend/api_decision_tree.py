from fastapi import APIRouter
from fastapi.responses import JSONResponse
import os
import json

router = APIRouter()

@router.get("/decision_tree")
async def get_decision_tree():
    # Adjust file path to absolute path for better reliability
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../descion_tree.json"))
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(content={"error": "Failed to load decision tree"}, status_code=500)
