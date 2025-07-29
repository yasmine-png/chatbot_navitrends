# Navitrends AI Assistant Project

## Overview
This project is an AI-powered chatbot assistant designed to help users navigate and inquire about Navitrends' professional digital transformation services. It features a decision tree chatbot with fuzzy matching capabilities and a fallback retrieval-augmented generation (RAG) system.

## Technologies Used

### Frontend
- React 18 with TypeScript
- Vite as the build tool
- Tailwind CSS for styling
- Radix UI components for accessible UI elements
- Framer Motion for animations

### Backend
- Python with FastAPI for API server
- Uvicorn as ASGI server
- Sentence-Transformers and FAISS for semantic search and embeddings
- FuzzyWuzzy and python-Levenshtein for fuzzy string matching
- Redis for caching and data storage, including conversation flow persistence and retrieval
- Aiosmtplib for asynchronous email sending
- Local Llama model (mistral-7b-instruct) for response generation
- Retrieval-Augmented Generation (RAG) pipeline combining FAISS and Llama
- Decision tree for guided conversation flow

## Features
- Decision tree chatbot with dynamic choices rendered as buttons
- Fuzzy matching of user questions to decision tree nodes
- Fallback to RAG + Llama for unmatched queries
- Interactive frontend with typing indicators and smooth UI
- Contact form integration within chatbot conversation

## Setup Instructions

### Backend
1. Create and activate a Python virtual environment.
2. Install dependencies from `backend/requirements.txt`.
3. Run the FastAPI server using Uvicorn.

### Frontend
1. Install Node.js dependencies using `npm install`.
2. Run the development server with `npm run dev`.

## Usage
- Interact with the chatbot via the web frontend.
- Select options from the decision tree or type questions.
- Contact support via the integrated contact form.

## Testing
- Manual testing of chatbot UI and backend API endpoints is recommended.
- Further automated tests can be added for comprehensive coverage.
- Critical areas tested include:
  - Frontend UI rendering and interaction for choice buttons
  - Backend API fuzzy matching and response correctness
  - Redis integration for storing and retrieving conversation history, including conversation flow persistence and retrieval
  - Edge cases and error handling

## Author
Yasmine Boukraiem - Expert in Digital Solutions & AI
