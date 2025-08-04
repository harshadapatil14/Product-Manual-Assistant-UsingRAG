"""
Configuration settings for Product Manual Assistant
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# AI Model Settings
DEFAULT_AI_MODEL = "gemini"
DEFAULT_LANGUAGE = "english"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100

# Supported Languages
SUPPORTED_LANGUAGES = {
    "english": {
        "name": "🇺🇸 English",
        "system": "You are a helpful assistant for a product manual. Use the context below to answer the question with accurate, complete detail.",
        "instructions": "- Break your answer into clear step-by-step instructions.\n- If a step involves safety or caution, mention it.\n- If the information is unavailable, say 'Not found in the document.'",
        "not_found": "Sorry, I couldn't find relevant information in the document."
    },
    "hindi": {
        "name": "🇮🇳 हिंदी (Hindi)",
        "system": "आप एक उत्पाद मैनुअल के लिए सहायक हैं। नीचे दिए गए संदर्भ का उपयोग करके सटीक और पूर्ण विवरण के साथ प्रश्न का उत्तर दें।",
        "instructions": "- अपने उत्तर को स्पष्ट चरण-दर-चरण निर्देशों में तोड़ें।\n- यदि किसी चरण में सुरक्षा या सावधानी शामिल है, तो उसका उल्लेख करें।\n- यदि जानकारी उपलब्ध नहीं है, तो कहें 'दस्तावेज़ में नहीं मिला।'",
        "not_found": "क्षमा करें, मुझे दस्तावेज़ में प्रासंगिक जानकारी नहीं मिली।"
    },
    "marathi": {
        "name": "🇮🇳 मराठी (Marathi)",
        "system": "तुम्ही उत्पाद मॅन्युअलसाठी सहाय्यक आहात. खाली दिलेल्या संदर्भाचा वापर करून प्रश्नाचे अचूक आणि पूर्ण तपशीलासह उत्तर द्या.",
        "instructions": "- तुमचे उत्तर स्पष्ट पायऱ्यांमध्ये विभाजित करा.\n- जर एखाद्या पायरीमध्ये सुरक्षा किंवा काळजी असेल तर त्याचा उल्लेख करा.\n- जर माहिती उपलब्ध नसेल तर 'दस्तऐवजात सापडले नाही' असे सांगा.",
        "not_found": "माफ करा, मला दस्तऐवजात संबंधित माहिती सापडली नाही."
    },
    "gujarati": {
        "name": "🇮🇳 ગુજરાતી (Gujarati)",
        "system": "તમે પ્રોડક્ટ મેન્યુઅલ માટે સહાયક છો. નીચે આપેલા સંદર્ભનો ઉપયોગ કરીને પ્રશ્નનો જવાબ ચોક્કસ અને સંપૂર્ણ વિગતો સાથે આપો.",
        "instructions": "- તમારા જવાબને સ્પષ્ટ પગલાં-દર-પગલાં સૂચનોમાં વિભાજિત કરો.\n- જો કોઈ પગલામાં સલામતી અથવા સાવધાની સમાવેશ થાય છે, તો તેનો ઉલ્લેખ કરો.\n- જો માહિતી ઉપલબ્ધ નથી, તો કહો 'દસ્તાવેજમાં મળ્યું નથી.'",
        "not_found": "માફ કરો, મને દસ્તાવેજમાં સંબંધિત માહિતી મળી નથી."
    },
    "bengali": {
        "name": "🇧🇩 বাংলা (Bengali)",
        "system": "আপনি একটি পণ্য ম্যানুয়ালের জন্য সহায়ক। নীচে দেওয়া প্রসঙ্গ ব্যবহার করে প্রশ্নের উত্তর সঠিক এবং সম্পূর্ণ বিবরণ সহ দিন।",
        "instructions": "- আপনার উত্তরকে স্পষ্ট ধাপে-ধাপে নির্দেশাবলীতে ভাগ করুন।\n- যদি কোন ধাপে নিরাপত্তা বা সতর্কতা জড়িত থাকে, তবে উল্লেখ করুন।\n- যদি তথ্য অনুপলব্ধ হয়, তবে বলুন 'নথিতে পাওয়া যায়নি।'",
        "not_found": "দুঃখিত, আমি নথিতে প্রাসঙ্গিক তথ্য খুঁজে পাইনি।"
    }
}

# AI Models Configuration
AI_MODELS = {
    "gemini": {
        "name": "🤖 Google Gemini Pro",
        "model_name": "gemini-1.5-pro"
    },
    "ollama": {
        "name": "🏠 Local Ollama (Llama3)",
        "model_name": "llama3"
    }
}

# Audio Settings
AUDIO_SETTINGS = {
    "default_rate": 150,
    "default_volume": 0.9,
    "fast_rate": 200,
    "timeout": 5,
    "phrase_time_limit": 10
}

# File Paths
DEFAULT_PERSIST_DIRECTORY = "data"
SESSION_DATA_DIR = "session_data"
LOGS_DIR = "logs"

# Vector Store Settings
VECTOR_STORE_SETTINGS = {
    "collection_name": "manual_chunks",
    "embedding_model": "all-MiniLM-L6-v2"
} 