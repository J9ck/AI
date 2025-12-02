"""
Hugging Face Transformers Examples
==================================

This module demonstrates how to use the Hugging Face Transformers library
for various NLP tasks.

Topics covered:
- Pipeline API for quick inference
- Loading models and tokenizers
- Fine-tuning BERT for classification
- Text generation with GPT-2
"""

# Note: Requires: pip install transformers torch

def sentiment_analysis_example():
    """
    Sentiment Analysis using Transformers Pipeline.
    
    The pipeline API provides a high-level interface for common NLP tasks.
    """
    
    print("=" * 60)
    print("SENTIMENT ANALYSIS")
    print("=" * 60)
    
    try:
        from transformers import pipeline
        
        # Create sentiment analysis pipeline
        # This loads a pre-trained model automatically
        classifier = pipeline("sentiment-analysis")
        
        # Example texts
        texts = [
            "I absolutely loved this movie! The acting was superb.",
            "This product is terrible. Complete waste of money.",
            "It was okay, nothing special but not bad either.",
            "The customer service was excellent and very helpful!",
        ]
        
        print("\nAnalyzing sentiments...\n")
        
        for text in texts:
            result = classifier(text)[0]
            print(f"Text: \"{text[:50]}...\"" if len(text) > 50 else f"Text: \"{text}\"")
            print(f"  → {result['label']}: {result['score']:.4f}")
            print()
            
    except ImportError:
        print("Note: transformers library not installed.")
        print("Install with: pip install transformers torch")
        print("\nExample output would be:")
        print('Text: "I absolutely loved this movie!"')
        print('  → POSITIVE: 0.9998')


def text_classification_manual():
    """
    Text Classification with manual model/tokenizer loading.
    
    This shows the lower-level API for more control.
    """
    
    print("=" * 60)
    print("MANUAL MODEL LOADING")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        # Load model and tokenizer
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        
        print(f"\nLoading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Set to evaluation mode
        model.eval()
        
        # Example text
        text = "This is the best day of my life!"
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        print(f"\nInput text: \"{text}\"")
        print(f"Tokenized: {inputs['input_ids'].tolist()}")
        print(f"Decoded tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        
        # Get prediction
        predicted_class = logits.argmax().item()
        confidence = probs[0][predicted_class].item()
        
        labels = ['NEGATIVE', 'POSITIVE']
        print(f"\nPrediction: {labels[predicted_class]} ({confidence:.4f})")
        
    except ImportError:
        print("Note: transformers library not installed.")


def named_entity_recognition_example():
    """
    Named Entity Recognition (NER) using Transformers.
    """
    
    print("=" * 60)
    print("NAMED ENTITY RECOGNITION")
    print("=" * 60)
    
    try:
        from transformers import pipeline
        
        # Create NER pipeline
        ner = pipeline("ner", aggregation_strategy="simple")
        
        # Example text
        text = "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976."
        
        print(f"\nText: \"{text}\"")
        print("\nEntities found:")
        
        entities = ner(text)
        for entity in entities:
            print(f"  • {entity['word']}")
            print(f"    Type: {entity['entity_group']}")
            print(f"    Score: {entity['score']:.4f}")
            print()
            
    except ImportError:
        print("Note: transformers library not installed.")
        print("\nExample output would be:")
        print("  • Apple Inc. (ORG)")
        print("  • Steve Jobs (PER)")
        print("  • Cupertino, California (LOC)")


def question_answering_example():
    """
    Question Answering using Transformers.
    """
    
    print("=" * 60)
    print("QUESTION ANSWERING")
    print("=" * 60)
    
    try:
        from transformers import pipeline
        
        # Create QA pipeline
        qa = pipeline("question-answering")
        
        # Context and questions
        context = """
        The Transformer architecture was introduced in the paper "Attention Is All You Need"
        by Vaswani et al. in 2017. It revolutionized natural language processing by using
        self-attention mechanisms instead of recurrence. BERT and GPT are both based on
        the Transformer architecture. BERT uses the encoder, while GPT uses the decoder.
        """
        
        questions = [
            "When was the Transformer introduced?",
            "What paper introduced the Transformer?",
            "What does BERT use?",
        ]
        
        print(f"\nContext: {context.strip()[:100]}...")
        print("\n" + "-" * 40)
        
        for question in questions:
            result = qa(question=question, context=context)
            print(f"\nQ: {question}")
            print(f"A: {result['answer']} (score: {result['score']:.4f})")
            
    except ImportError:
        print("Note: transformers library not installed.")


def text_generation_example():
    """
    Text Generation using GPT-2.
    """
    
    print("=" * 60)
    print("TEXT GENERATION (GPT-2)")
    print("=" * 60)
    
    try:
        from transformers import pipeline
        
        # Create text generation pipeline
        generator = pipeline("text-generation", model="gpt2")
        
        # Prompt
        prompt = "Artificial intelligence will"
        
        print(f"\nPrompt: \"{prompt}\"")
        print("\nGenerated continuations:")
        print("-" * 40)
        
        # Generate multiple completions
        outputs = generator(
            prompt,
            max_length=50,
            num_return_sequences=2,
            do_sample=True,
            temperature=0.7
        )
        
        for i, output in enumerate(outputs, 1):
            print(f"\n{i}. {output['generated_text']}")
            
    except ImportError:
        print("Note: transformers library not installed.")


def summarization_example():
    """
    Text Summarization using Transformers.
    """
    
    print("=" * 60)
    print("TEXT SUMMARIZATION")
    print("=" * 60)
    
    try:
        from transformers import pipeline
        
        # Create summarization pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Long text to summarize
        article = """
        Machine learning is a subset of artificial intelligence that provides systems the 
        ability to automatically learn and improve from experience without being explicitly 
        programmed. Machine learning focuses on the development of computer programs that 
        can access data and use it to learn for themselves. The process of learning begins 
        with observations or data, such as examples, direct experience, or instruction, in 
        order to look for patterns in data and make better decisions in the future based on 
        the examples that we provide. The primary aim is to allow the computers to learn 
        automatically without human intervention or assistance and adjust actions accordingly.
        """
        
        print(f"\nOriginal ({len(article.split())} words):")
        print(article.strip())
        
        summary = summarizer(article, max_length=50, min_length=20, do_sample=False)
        
        print(f"\nSummary ({len(summary[0]['summary_text'].split())} words):")
        print(summary[0]['summary_text'])
        
    except ImportError:
        print("Note: transformers library not installed.")


def embedding_example():
    """
    Getting embeddings from transformer models.
    """
    
    print("=" * 60)
    print("TEXT EMBEDDINGS")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        # Load model
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Texts to embed
        texts = [
            "I love machine learning",
            "AI and ML are fascinating fields",
            "The weather is nice today",
        ]
        
        print("\nComputing embeddings...")
        
        def get_embedding(text):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            # Mean pooling
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            return embedding
        
        embeddings = [get_embedding(text) for text in texts]
        
        # Compute cosine similarity
        def cosine_similarity(a, b):
            return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))
        
        print("\nCosine Similarities:")
        print("-" * 40)
        
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = cosine_similarity(embeddings[i], embeddings[j]).item()
                print(f"'{texts[i][:30]}...' vs '{texts[j][:30]}...'")
                print(f"  Similarity: {sim:.4f}")
                print()
                
    except ImportError:
        print("Note: transformers library not installed.")


def demo_all():
    """Run all examples."""
    
    print("\n" + "=" * 60)
    print("HUGGING FACE TRANSFORMERS EXAMPLES")
    print("=" * 60)
    
    examples = [
        ("Sentiment Analysis", sentiment_analysis_example),
        ("Manual Model Loading", text_classification_manual),
        ("Named Entity Recognition", named_entity_recognition_example),
        ("Question Answering", question_answering_example),
        ("Text Generation", text_generation_example),
        # ("Summarization", summarization_example),  # Requires large model
        # ("Embeddings", embedding_example),  # Requires additional model
    ]
    
    for name, func in examples:
        print(f"\n\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}\n")
        
        try:
            func()
        except Exception as e:
            print(f"Error: {e}")
        
        print()


if __name__ == "__main__":
    # Run individual examples or all
    sentiment_analysis_example()
