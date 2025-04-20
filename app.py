from routes.automation import router as automation_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

# Create the FastAPI app
app = FastAPI(title="Automation Extension Backend",
              description="Backend for processing natural language automation requests",
              version="1.0.0")

# Add CORS middleware to allow requests from the browser extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - change this in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Import and include routers
app.include_router(automation_router, prefix="")

# Health check endpoint


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "automation-extension-backend"}

# Run the server when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=False)
