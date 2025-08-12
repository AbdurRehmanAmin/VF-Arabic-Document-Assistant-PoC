# app.py
import chainlit as cl
from chainlit.types import AskFileResponse
from typing import List, Dict, Optional, Any
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from document_processor import process_document, chunk_document, detect_language
from vector_store import VectorStore
from ollama_service import generate_starters, query_document, translate_text

# Configuration
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize vector store at module level to make it accessible across handlers
vector_store: Optional[VectorStore] = None

@cl.on_chat_start
async def start():
    """Initialize the chat session and welcome the user"""
    global vector_store
    
    # Initialize vector store
    vector_store = VectorStore()
    
    # Initialize session variables
    cl.user_session.set("files", [])
    cl.user_session.set("vector_store", vector_store)
    cl.user_session.set("processed_files", 0)  # Track number of processed files
    
    # Welcome message with file upload instructions
    await cl.Message(
        content="""# مرحبًا بك في مساعد المستندات العربية! 👋

أنا مساعدك الافتراضي المتخصص في تحليل المستندات والإجابة على أسئلتك. يمكنني العمل مع المستندات باللغة العربية والإنجليزية.

قم بتحميل مستند للبدء. يمكنك تحميل ما يصل إلى 5 مستندات في كل محادثة.
        """,
        author="المساعد السعودي للمستندات"
    ).send()
    
    # Request file upload
    files = await cl.AskFileMessage(
        content="يرجى تحميل ملف PDF أو DOCX أو TXT للبدء:",
        accept=["application/pdf", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
        max_size_mb=20,
        max_files=5,  # Allow up to 5 files at once
        timeout=180,
        author="المساعد السعودي للمستندات"
    ).send()
    
    # Process the uploaded files if any
    if files:
        for file in files:
            await process_uploaded_file(file)

async def process_uploaded_file(file: AskFileResponse):
    """Process an uploaded file"""
    global vector_store
    
    # Check if vector_store is initialized
    if vector_store is None:
        vector_store = VectorStore()
        cl.user_session.set("vector_store", vector_store)
    
    # Get processed files count with a default value
    processed_files = cl.user_session.get("processed_files")
    processed_files = 0 if processed_files is None else processed_files
    
    # Check if we've reached the file limit (5 per chat)
    if processed_files >= 5:
        await cl.Message(
            content="تم الوصول إلى الحد الأقصى لعدد الملفات (5 ملفات). يرجى بدء محادثة جديدة لتحميل المزيد من الملفات.",
            author="المساعد السعودي للمستندات"
        ).send()
        return
    
    # Show processing message with steps
    msg = cl.Message(
        content=f"جاري معالجة `{file.name}`... (الخطوة 1/4: حفظ الملف)",
        author="المساعد السعودي للمستندات"
    )
    await msg.send()
    
    try:
        # Save the file - use file.path which is already available
        file_path = file.path
        
        # Store file reference in session
        files = cl.user_session.get("files")
        if files is None:
            files = []
        
        files.append({"name": file.name, "path": file_path})
        cl.user_session.set("files", files)
        
        # Update processed files count
        cl.user_session.set("processed_files", processed_files + 1)
        
        # Update progress
        msg.content = f"جاري معالجة `{file.name}`... (الخطوة 2/4: استخراج النص)"
        await msg.update()
        
        # Process the document
        document_content = await process_document(file_path)
        
        # Check if document content was extracted successfully
        if not document_content or len(document_content.strip()) == 0:
            msg.content = f"⚠️ لم يتم استخراج أي نص من الملف `{file.name}`. يرجى التأكد من أن الملف يحتوي على نص قابل للاستخراج."
            await msg.update()
            return
            
        logger.info(f"Extracted {len(document_content)} characters from document")
        
        # Detect language
        language = detect_language(document_content)
        cl.user_session.set("document_language", language)
        
        # Update progress
        msg.content = f"جاري معالجة `{file.name}`... (الخطوة 3/4: إنشاء فهرس دلالي)"
        await msg.update()
        
        # Chunk the document using semantic chunking
        chunks = await chunk_document(document_content)
        
        if not chunks:
            msg.content = f"⚠️ لم يتم إنشاء مقتطفات من الملف `{file.name}`. يرجى التأكد من أن الملف يحتوي على نص."
            await msg.update()
            return
            
        logger.info(f"Created {len(chunks)} chunks from document")
        
        # Ensure vector_store is available
        if vector_store is None:
            vector_store = VectorStore()
            cl.user_session.set("vector_store", vector_store)
            
        # Store in vector database
        await vector_store.add_document(chunks)
        cl.user_session.set("vector_store", vector_store)
        
        # Update progress
        msg.content = f"جاري معالجة `{file.name}`... (الخطوة 4/4: إنشاء أسئلة مقترحة)"
        await msg.update()
        
        # Generate starters with LLM
        starters = await generate_starters(document_content)
        
        # Update processing message
        language_text = "العربية" if language == "arabic" else "الإنجليزية"
        language_emoji = "🇸🇦" if language == "arabic" else "🇬🇧"
        
        # Create document preview element
        preview_text = document_content[:500] + "..." if len(document_content) > 500 else document_content
        
        # Update final message
        processed_files = cl.user_session.get("processed_files")
        processed_files = 0 if processed_files is None else processed_files
        file_count_text = f"({processed_files}/5)"
        
        # Update message content first
        msg.content = f"✅ تم تحميل الملف: `{file.name}` ({language_emoji} {language_text}) {file_count_text}\n\nتمت معالجة المستند الخاص بك ويمكنك الآن طرح أسئلة حوله!"
        await msg.update()
        
        # Send document preview as a separate message
        preview_element = cl.Text(name=file.name, content=preview_text)
        await cl.Message(
            content="معاينة المستند:", 
            elements=[preview_element], 
            author="المساعد السعودي للمستندات"
        ).send()
        
        # Set suggested questions as starters
        if starters:
            starter_message = "إليك بعض الأسئلة التي قد ترغب في طرحها:"
            await cl.Message(content=starter_message, author="المساعد السعودي للمستندات").send()
            
            for starter in starters:
                # Send each starter question with an action button
                starter_msg = cl.Message(
                    content=starter, 
                    author="اقتراحات"
                )
                await starter_msg.send()
                
                # Send an action button as a separate message
                action_msg = cl.Message(
                    content="",
                    actions=[
                        cl.Action(
                            name="ask_question",
                            label="اسأل هذا السؤال",
                            value=starter,
                            payload={"question": starter}
                        )
                    ],
                    author="اقتراحات"
                )
                await action_msg.send()
        
        # Offer to upload more files if under the limit
        processed_files = cl.user_session.get("processed_files")
        processed_files = 0 if processed_files is None else processed_files
        
        if processed_files < 5:
            await cl.Message(
                content="هل ترغب في تحميل المزيد من الملفات؟ يمكنك تحميل ما يصل إلى 5 ملفات في كل محادثة.",
                author="المساعد السعودي للمستندات",
                actions=[
                    cl.Action(
                        name="upload_more", 
                        label="تحميل المزيد من الملفات",
                        payload={"action": "upload_more"}
                    )
                ]
            ).send()
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        # Update error message
        msg.content = f"⚠️ خطأ في معالجة الملف: {str(e)}"
        await msg.update()

@cl.action_callback("ask_question")
async def on_ask_question(action):
    """Handle asking a suggested question"""
    question = action.payload.get("question", "")
    if question:
        await cl.Message(content=question).send()
        await handle_user_message(question)

@cl.action_callback("translate_question")
async def on_translate_text(action):
    """Handle translating text"""
    text = action.payload.get("text", "")
    target = action.payload.get("target", "english")
    
    if text:
        translated = await translate_text(text, target)
        
        # Determine source and target language names
        source_lang = "العربية" if any('\u0600' <= c <= '\u06FF' for c in text) else "الإنجليزية"
        target_lang = "الإنجليزية" if target == "english" else "العربية"
        
        await cl.Message(
            content=f"**الترجمة من {source_lang} إلى {target_lang}:**\n\n{translated}",
            author="الترجمة"
        ).send()

@cl.action_callback("upload_more")
async def on_upload_more(action):
    """Handle the upload more action"""
    # Request additional file upload
    files = await cl.AskFileMessage(
        content="يرجى تحميل ملف إضافي:",
        accept=["application/pdf", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
        max_size_mb=20,
        max_files=1,
        timeout=180,
        author="المساعد السعودي للمستندات"
    ).send()
    
    # Process the uploaded file if any
    if files and len(files) > 0:
        await process_uploaded_file(files[0])

@cl.action_callback("translate_response")
async def on_translate_response(action):
    """Handle translating a response"""
    text = action.payload.get("text", "")
    target = action.payload.get("target", "english")
    
    if text:
        translated = await translate_text(text, target)
        
        # Determine source and target language names
        source_lang = "العربية" if any('\u0600' <= c <= '\u06FF' for c in text) else "الإنجليزية"
        target_lang = "الإنجليزية" if target == "english" else "العربية"
        
        await cl.Message(
            content=f"**الترجمة من {source_lang} إلى {target_lang}:**\n\n{translated}",
            author="الترجمة"
        ).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages"""
    await handle_user_message(message.content)

async def handle_user_message(user_message: str):
    """Process user message and generate response"""
    global vector_store
    
    # Ensure vector store is available
    if vector_store is None or not hasattr(vector_store, 'index') or vector_store.index is None:
        await cl.Message(
            content="يرجى تحميل مستند أولاً.",
            author="المساعد السعودي للمستندات"
        ).send()
        return
    
    try:
        # Create a temporary message to show processing status
        processing_msg = cl.Message(
            content="🔄 جاري معالجة سؤالك...",
            author="المساعد السعودي للمستندات"
        )
        await processing_msg.send()
        
        try:
            # Search for relevant chunks
            relevant_chunks = await vector_store.search(user_message)
            
            if not relevant_chunks:
                # No results found
                processing_msg.content = "لم أتمكن من العثور على معلومات ذات صلة في المستندات للإجابة على سؤالك."
                await processing_msg.update()
                return
                
            # Log the chunks for debugging
            logger.info(f"Found {len(relevant_chunks)} relevant chunks")
            for i, chunk in enumerate(relevant_chunks):
                logger.info(f"Chunk {i+1} (score: {chunk['score']}): {chunk['chunk'][:100]}...")
                
            # Create a compact references list for the response
            reference_list = []
            for i, chunk in enumerate(relevant_chunks):
                metadata = chunk.get("metadata", {})
                page_num = metadata.get("page", "?")
                line_num = metadata.get("line", "?")
                ref_text = f"[مرجع {i+1}] صفحة {page_num}, سطر {line_num}"
                reference_list.append(ref_text)
                
            # Query Ollama LLM with the user question and relevant chunks
            response = await query_document(user_message, relevant_chunks)
            
            # Append references to the response if they're not already included
            if "مرجع" not in response and "المراجع" not in response:
                references_text = "\n\n**المراجع:**\n" + "\n".join(reference_list)
                logging.info(f"Appending references to response: {references_text}")
                response += references_text
                
            # Update the response message
            processing_msg.content = response
            await processing_msg.update()
            
            # Create action for translation in a separate message
            translate_action = cl.Action(
                name="translate_response",
                label="ترجم إلى الإنجليزية",
                value=response,
                payload={"text": response, "target": "english"}
            )
            
            # Send translation action as separate message
            await cl.Message(
                content="",
                actions=[translate_action],
                author="المساعد السعودي للمستندات"
            ).send()
            
            # Create elements for sources/citations with improved references
            elements = []
            for references_text in reference_list:
                elements.append(
                    cl.Text(
                        name=references_text,
                        content=references_text,
                        display="inline"  # Display inline in the message
                    )
                )
                '''
            for i, chunk in enumerate(relevant_chunks):
                # Get metadata for display
                metadata = chunk.get("metadata", {})
                page_num = metadata.get("page", "?")
                line_num = metadata.get("line", "?")
                
                # Create a more informative name for the source
                ref_name = f"مرجع {i+1} (صفحة {page_num}, سطر {line_num})"
                
                # Create a citation element for each chunk
                chunk_text = chunk["chunk"]
                
                elements.append(
                    cl.Text(
                        name=ref_name,
                        content=chunk_text,
                        display="side"  # Display as a sidebar element
                    )
                )'''
            
            sources = "المصادر المستخدمة في الإجابة:" + '\n'.join(reference_list[0:])
            logging.info(f"Sources for response: {sources}")
            # Send source elements as a separate message
            if elements:
                await cl.Message(
                    content= sources,
                    elements=elements,
                    author="المساعد السعودي للمستندات"
                ).send()
                
        except Exception as e:
            logger.error(f"Error during search or query: {str(e)}")
            processing_msg.content = f"حدث خطأ أثناء البحث: {str(e)}"
            await processing_msg.update()
    
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        await cl.Message(
            content=f"حدث خطأ: {str(e)}",
            author="المساعد السعودي للمستندات"
        ).send()