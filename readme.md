# Milvus Vector Store – PDF to Embeddings

This project demonstrates how to store PDF content as vector embeddings in Milvus using Python.

---

## Prerequisites

- Docker
- Python 3.9+
- OpenAI API key

---

## Setup

Clone the repository:

git clone https://github.com/prince0018/Milvus-Vector-Store.git  
cd Milvus-Vector-Store  

Start Milvus using Docker:

```curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh  ```

```bash standalone_embed.sh start  ```

Milvus will be available at localhost:19530

---

Create Python virtual environment:

python -m venv .venv  
source .venv/bin/activate  
pip install -r requirements.txt  

---

Configure environment variables:

touch .env  

Add your OpenAI API key to `.env`:

OPENAI_API_KEY=your_openai_api_key_here

---

## Usage

1. Place your PDF file inside the `data/` directory  
2. Update the PDF path in `src/index.py` if needed  
3. Run the script:

python src/index.py  

Expected output:

PDF embeddings stored in Milvus

---

## Notes

- Milvus collection is equivalent to an Elasticsearch index  
- Milvus stores text chunks, not entire PDFs  
- Docker runtime files are ignored via `.gitignore`

---

Stop Milvus:

```bash standalone_embed.sh stop```