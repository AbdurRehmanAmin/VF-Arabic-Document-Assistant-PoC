# ollama_service.py
import aiohttp
import logging
from typing import List, Dict, Any
import asyncio
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama API configuration - defaults to local deployment
OLLAMA_API_URL = "http://localhost:11434/api"
OLLAMA_MODEL = "llama3"

# Saudi expert system prompt
SAUDI_EXPERT_SYSTEM_PROMPT = """
أنت خبير سعودي متخصص في المساعدة وتحليل المستندات باللغة العربية. تستخدم لغة عربية فصيحة وواضحة، وتقدم إجابات دقيقة ومفيدة. تعتبر نفسك مستشارًا سعوديًا محترفًا وتتواصل بأسلوب مهذب ومحترم.

قواعد الإجابة:
1. استخدم فقط المعلومات الموجودة في السياق المقدم للإجابة.
2. إذا كانت المعلومات غير متوفرة في السياق، قل "لا تتوفر لدي معلومات كافية للإجابة على هذا السؤال."
3. الإجابة تكون دائمًا باللغة العربية الفصحى بغض النظر عن لغة السؤال أو المستند.
4. عند الاقتباس من النص، استخدم عبارة "وفقًا للمرجع [رقم]" أو "كما ورد في المرجع [رقم]".
5. قدم إجابات مختصرة ومباشرة ومفيدة.
"""

async def generate_starters(document_content: str) -> List[str]:
    """Generate starter questions based on document content using Ollama."""
    logger.info(f"Generating starter questions using Ollama model: {OLLAMA_MODEL}")
    
    # Get a sample of the document to generate starters
    sample = document_content[:5000] if len(document_content) > 2000 else document_content
    
    prompt = f"""
    {SAUDI_EXPERT_SYSTEM_PROMPT}
    
    استنادًا إلى المستند التالي، قم بإنشاء 5 أسئلة محتملة قد يطرحها المستخدم. اكتب الأسئلة باللغة العربية فقط.
    
    المستند:
    {sample}
    
    قم بإنشاء 5 أسئلة فقط تتعلق بمحتوى المستند. اكتب الأسئلة باللغة العربية فقط، سطر واحد لكل سؤال:
    """
    
    try:
        response_text = await call_ollama(prompt)
        
        # Extract questions from response
        logger.info(response_text)
        questions = [q.strip() for q in response_text.split('\n') if q.strip() and '؟' in q]
        logger.info(f"Generated {len(questions)} starter questions")
        return questions[:5]  # Return up to 5 questions
    except Exception as e:
        logger.error(f"Error generating starter questions: {str(e)}")
        return []

async def query_document(query: str, context: List[Dict[str, Any]]) -> str:
    """Query Ollama with user question and document context."""
    logger.info(f"Querying Ollama model: {OLLAMA_MODEL} with document context")
    
    # Format context for the prompt with citations
    context_chunks = []
    for i, item in enumerate(context):
        # Add citation metadata
        chunk_text = item["chunk"]
        metadata = item.get("metadata", {})
        
        # Get page and line info if available
        page_num = metadata.get("page", "?")
        line_num = metadata.get("line", "?")
        
        # Format reference ID with page/line info
        ref_id = f"[مرجع {i+1} (صفحة {page_num}, سطر {line_num})]"
        
        # Add the chunk with its reference ID
        context_chunks.append(f"{ref_id}: {chunk_text}")
    
    context_text = "\n\n".join(context_chunks)
    
    prompt = f"""
    {SAUDI_EXPERT_SYSTEM_PROMPT}
    
    السياق (يتكون من مقتطفات من المستند):
    {context_text}
    
    السؤال:
    {query}
    
    قدم إجابة دقيقة مستندة فقط على المعلومات الواردة في السياق. عندما تستشهد بمعلومات، استخدم أرقام المراجع المقدمة مثل [مرجع 1] أو [مرجع 2] للإشارة إلى مصدر المعلومات. لا تكتب "وفقًا للنص" فقط، بل استخدم رقم المرجع دائمًا.
    الإجابة:
    """
    
    try:
        response_text = await call_ollama(prompt)
        return response_text
    except Exception as e:
        logger.error(f"Error querying document: {str(e)}")
        return f"حدث خطأ: {str(e)}"

async def call_ollama(prompt: str) -> str:
    """Call the Ollama API."""
    generate_url = f"{OLLAMA_API_URL}/generate"
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.5,  # Lower temperature for more factual responses
            "num_predict": 1024  # Increased token limit for better responses
        }
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(generate_url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "")
                else:
                    error_text = await response.text()
                    logger.error(f"Error from Ollama API: {response.status}, {error_text}")
                    return f"خطأ في الاتصال: {response.status}"
    except Exception as e:
        logger.error(f"Exception calling Ollama API: {str(e)}")
        return f"خطأ: {str(e)}"

async def translate_text(text: str, target_language: str) -> str:
    """Translate text using Ollama."""
    logger.info(f"Translating text to {target_language}")
    
    source_language = "Arabic" if any('\u0600' <= c <= '\u06FF' for c in text) else "English"
    target_language_name = "الإنجليزية" if target_language == "english" else "العربية"
    
    prompt = f"""
    ترجم النص التالي من {source_language} إلى {target_language_name}:
    
    النص:
    {text}
    
    الترجمة:
    """
    
    try:
        response_text = await call_ollama(prompt)
        return response_text
    except Exception as e:
        logger.error(f"Error translating text: {str(e)}")
        return f"خطأ في الترجمة: {str(e)}"