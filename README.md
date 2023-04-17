# auto-nlp-classification-gpt
This project aims to provide an autonomous pipeline for creating NLP models for text classification. It is based on GPT-3.5-Turbo, Ada-Embedding, and PyTorch.

# settings (add OPENAI_API_KEY)
```bash
    touch .env 
    OPENAI_API_KEY=sk-
```

## usage
```bash
    python -m venv env 
    source env/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt 

    python main.py -cls class0 -cls class1 -cls class2 ... -cls classN -limit 50 --port nb_epochs 64 --batch_size 16 -lr 0.001
    # go to http://localhost:8000/docs to use the server
```

