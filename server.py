from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os

# Import your function (replace with a dummy function for testing)
def get_recommendations(query):
    return f"Echo: {query}"

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Serve HTML
@app.get("/", response_class=HTMLResponse)
def get_chat():
    with open("frontend/chat.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
@app.get("/", response_class=HTMLResponse)
def get_chat():
    html_content = "<h1>Test page loaded!</h1>"
    return HTMLResponse(content=html_content)


# Chat endpoint
@app.post("/chat")
async def chat_endpoint(user_input: str = Form(...)):
    response_text = get_recommendations(user_input)
    return JSONResponse({"response": response_text})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
