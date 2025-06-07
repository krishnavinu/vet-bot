import streamlit as st
from agent import diagnose_symptoms
import io
import numpy as np
from PIL import Image
from vetconnect import find_nearby_vets
import json
import os
from image_analysis import analyze_animal_image, get_health_assessment
import base64
import random

try:
    import sounddevice as sd
    import scipy.io.wavfile as wav
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    SOUNDDEVICE_AVAILABLE = False

def process_diagnosis_result(diagnosis, animal_type):
    """Display the diagnosis result in a formatted box"""
    # Create colored box based on confidence and condition
    if diagnosis["condition"].lower() == "unknown" or diagnosis["confidence"] < 30:
        box_color = "#d32f2f"  # Darker red for unknown/very low confidence
        text_color = "#ffffff"  # White text
        icon = "⚠️"
    elif diagnosis["needs_vet"]:
        box_color = "#ffdada"  # Lighter red background
        text_color = "#d32f2f"  # Darker red text
        icon = "🏥"
    else:
        box_color = "#deffde"  # Lighter green background
        text_color = "#1b5e20"  # Darker green text
        icon = "📋"
    
    st.markdown(f"""
    <div class="diagnosis-box" style="background-color: {box_color};">
        <h3 style="color: {text_color};">{icon} Diagnosis Result</h3>
        <p style="color: {text_color if diagnosis['condition'].lower() == 'unknown' else '#000000'}">
            <strong>Animal Type:</strong> {animal_type}</p>
        <p style="color: {text_color if diagnosis['condition'].lower() == 'unknown' else '#000000'}">
            <strong>Condition:</strong> {diagnosis["condition"].title()}</p>
        <p style="color: {text_color if diagnosis['condition'].lower() == 'unknown' else '#000000'}">
            <strong>Confidence Level:</strong> {diagnosis["confidence"]}%</p>
        <p style="color: {text_color if diagnosis['condition'].lower() == 'unknown' else '#000000'}">
            <strong>{"⚠️ Professional Veterinary Care Needed:" if diagnosis["needs_vet"] else "📋 Recommended Treatment:"}</strong></p>
        <p style="color: {text_color if diagnosis['condition'].lower() == 'unknown' else '#000000'}; white-space: pre-line">
            {diagnosis["recommendation"]}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show vet finder prompt if veterinary care is needed
    if diagnosis["needs_vet"]:
        st.info("👉 Use the 'Find Veterinarian' tab to locate a vet in your area.")

def process_diagnosis(input_data, input_type, animal_type):
    """Process diagnosis with any input type and show results"""
    with st.spinner("Analyzing symptoms..."):
        try:
            diagnosis = diagnose_symptoms(
                input_data,
                input_type=input_type,
                animal_type=animal_type
            )
            process_diagnosis_result(diagnosis, animal_type)
        except Exception as e:
            st.error(f"An error occurred during diagnosis: {str(e)}")

# Load follow-up questions
def load_follow_up_questions():
    questions_path = os.path.join("data", "follow_up_questions.json")
    with open(questions_path, 'r') as f:
        return json.load(f)

# Function to get relevant follow-up questions
def get_follow_up_questions(animal_type, symptoms):
    questions = load_follow_up_questions()
    all_questions = []
    
    # Add general questions to pool
    all_questions.extend(questions['general_questions']['eating_habits'])
    all_questions.extend(questions['general_questions']['behavior'])
    
    # Add animal-specific questions to pool
    animal_type = animal_type.lower()
    if animal_type in questions['animal_specific']:
        for category in questions['animal_specific'][animal_type]:
            all_questions.extend(questions['animal_specific'][animal_type][category])
    
    # Add emergency questions if certain keywords are present
    emergency_keywords = ['severe', 'emergency', 'critical', 'dying', 'collapsed', 'bleeding']
    if any(keyword in symptoms.lower() for keyword in emergency_keywords):
        all_questions.extend(questions['emergency_indicators']['severe_symptoms'])
    
    # Randomly select 5 unique questions
    if len(all_questions) > 5:
        selected_questions = random.sample(all_questions, 5)
    else:
        selected_questions = all_questions
    
    return selected_questions

# Set page configuration
st.set_page_config(
    page_title="Bio Guardian - Livestock Health Diagnosis",
    page_icon="🐮",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        color: #2e7d32;
        text-align: center;
        padding: 20px;
        font-size: 3em;
        font-weight: bold;
    }
    .subtitle {
        color: #1b5e20;
        text-align: center;
        padding-bottom: 30px;
    }
    .diagnosis-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .symptom-guide {
        background-color: #1b5e20s;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .vet-card {
        background-color: #d32f2f;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown("<h1 class='main-title'>🐮 Bio Guardian</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Advanced Livestock Health Diagnosis System</p>", unsafe_allow_html=True)

# Initialize session state for diagnosis flow
if 'diagnosis_state' not in st.session_state:
    st.session_state.diagnosis_state = {
        'initial_description': '',
        'asking_questions': False,
        'current_questions': [],
        'answers': {},
        'final_diagnosis': None
    }

# Initialize session state for follow-up questions
if 'follow_up_state' not in st.session_state:
    st.session_state.follow_up_state = {
        'asking_questions': False,
        'current_questions': [],
        'answers': {},
        'original_symptoms': ''
    }

# Add info about the services in the sidebar
st.sidebar.markdown("### 🤖 Powered by")
st.sidebar.markdown("""
- 🖼️ Clarifai - Image Analysis
- 🗣️ Google Speech-to-Text (Free Tier)
- 🧠 Groq - AI Diagnosis
""")

# Create tabs
tab1, tab2 = st.tabs(["Diagnose Symptoms", "Find Veterinarian"])

# Diagnosis Tab
with tab1:
    st.header("🩺 Symptom Diagnosis")
    
    # Animal type selection
    animal_type = st.selectbox(
        "Select Animal Type",
        options=["Cattle", "Buffalo", "Sheep", "Goat", "Poultry"],
        key="diagnosis_animal"
    )
    
    # Input method selection
    input_method = st.radio(
        "Choose how to describe symptoms",
        options=["Text", "Voice", "Image"],
        horizontal=True
    )
    
    if input_method == "Text":
        # Symptom guide
        st.markdown("""
        <div class="symptom-guide">
            <h4>📝 How to Describe Symptoms:</h4>
            <ul>
                <li>What changes have you noticed in the animal?</li>
                <li>When did the symptoms start?</li>
                <li>Has there been any change in eating or drinking?</li>
                <li>Are there any visible signs (swelling, wounds, etc.)?</li>
                <li>Has the animal's behavior changed?</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize session state for diagnosis flow
        if 'diagnosis_state' not in st.session_state:
            st.session_state.diagnosis_state = {
                'initial_description': '',
                'asking_questions': False,
                'current_questions': [],
                'answers': {},
                'final_diagnosis': None
            }
        
        symptoms = st.text_area(
            "Describe the Symptoms",
            value=st.session_state.diagnosis_state['initial_description'],
            placeholder=(
                f"Example: My {animal_type.lower()} has:\n"
                "- Reduced appetite since yesterday\n"
                "- Seems less active than usual\n"
                "- [Add other symptoms you've noticed]"
            ),
            height=150
        )
        
        # Store symptoms in session state when they change
        if symptoms != st.session_state.diagnosis_state['initial_description']:
            st.session_state.diagnosis_state['initial_description'] = symptoms
            st.session_state.diagnosis_state['asking_questions'] = False
            st.session_state.diagnosis_state['answers'] = {}
        
        analyze_button = st.button("Analyze Symptoms", key="text_diagnose")
        
        if analyze_button or st.session_state.diagnosis_state['asking_questions']:
            if symptoms:
                if not st.session_state.diagnosis_state['asking_questions']:
                    # First pass - analyze symptoms
                    initial_diagnosis = diagnose_symptoms(symptoms, "text", animal_type)
                    
                    if initial_diagnosis.get('needs_followup', False):
                        st.session_state.diagnosis_state['asking_questions'] = True
                        st.session_state.diagnosis_state['current_questions'] = get_follow_up_questions(animal_type, symptoms)
                        st.rerun()
                    else:
                        process_diagnosis_result(initial_diagnosis, animal_type)
                        st.session_state.diagnosis_state['asking_questions'] = False
                
                elif st.session_state.diagnosis_state['asking_questions']:
                    st.info("Please answer these questions to help us provide a more accurate diagnosis:")
                    
                    # Create a form for the questions
                    with st.form("follow_up_questions"):
                        answers = {}
                        for q in st.session_state.diagnosis_state['current_questions']:
                            st.write("---")
                            st.write(f"**{q}**")
                            answer = st.radio(
                                "Select your answer:",
                                options=["Yes", "No"],
                                key=f"q_{q}",
                                horizontal=True,
                                label_visibility="collapsed"
                            )
                            answers[q] = answer
                        
                        submitted = st.form_submit_button("Submit Answers")
                        
                        if submitted:
                            # Combine original symptoms with answers
                            detailed_symptoms = f"{symptoms}\n\nAdditional Information:\n"
                            for q, a in answers.items():
                                detailed_symptoms += f"- {q}: {a}\n"
                            
                            # Get final diagnosis
                            final_diagnosis = diagnose_symptoms(detailed_symptoms, "text", animal_type)
                            
                            # Reset state
                            st.session_state.diagnosis_state['asking_questions'] = False
                            st.session_state.diagnosis_state['current_questions'] = []
                            st.session_state.diagnosis_state['answers'] = {}
                            st.session_state.diagnosis_state['initial_description'] = ''
                            
                            # Show results
                            process_diagnosis_result(final_diagnosis, animal_type)
            else:
                st.warning("Please describe the symptoms first.")
    
    elif input_method == "Voice":
        if SOUNDDEVICE_AVAILABLE:
            st.markdown("""
            <div class="symptom-guide">
                <h4>🎤 Voice Recording Instructions:</h4>
                <ul>
                    <li>Click 'Start Recording' and speak clearly</li>
                    <li>Describe all symptoms you've noticed</li>
                    <li>Click 'Stop Recording' when finished</li>
                    <li>Maximum recording duration: 10 seconds</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Initialize session state variables
            if 'recording' not in st.session_state:
                st.session_state.recording = False
            if 'audio_data' not in st.session_state:
                st.session_state.audio_data = None
            if 'recording_started' not in st.session_state:
                st.session_state.recording_started = False
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Start Recording" if not st.session_state.recording else "Stop Recording"):
                    if not st.session_state.recording:  # Starting recording
                        st.session_state.recording = True
                        st.session_state.recording_started = True
                        st.session_state.audio_data = None
                        st.rerun()
                    else:  # Stopping recording
                        st.session_state.recording = False
                        st.rerun()
            
            with col2:
                if st.session_state.recording:
                    st.write(f"🔴 Recording in progress...")
                    
                    try:
                        # Record audio using sounddevice
                        duration = 10  # seconds
                        fs = 44100  # Sample rate
                        channels = 1
                        
                        # Show countdown
                        for remaining in range(duration, 0, -1):
                            st.write(f"Recording... {remaining} seconds remaining")
                            if remaining == duration:  # Only record on first iteration
                                recording = sd.rec(int(duration * fs), 
                                                samplerate=fs, 
                                                channels=channels, 
                                                dtype='int16',
                                                blocking=True)
                        
                        st.write("Processing recording...")
                        
                        # Create WAV file in memory
                        wav_buffer = io.BytesIO()
                        wav.write(wav_buffer, fs, recording)
                        wav_buffer.seek(0)
                        audio_data = wav_buffer.read()
                        
                        # Store in session state
                        st.session_state.audio_data = audio_data
                        st.session_state.recording = False
                        
                        # Process the recording
                        if audio_data is not None:
                            st.write("Analyzing recording...")
                            process_diagnosis(audio_data, "voice", animal_type)
                        else:
                            st.error("No audio data was recorded")
                        
                    except Exception as e:
                        st.error(f"Recording error: {str(e)}")
                        st.session_state.recording = False
                        st.session_state.audio_data = None
                
                elif st.session_state.recording_started and not st.session_state.recording:
                    if st.session_state.audio_data is not None:
                        st.success("Recording completed and processed!")
                    else:
                        st.warning("Recording failed or no audio was captured")
        else:
            st.warning("Audio recording is not supported on this platform. Please upload an audio file or use text input.")
    
    else:  # Image input
        st.markdown("""
        <div class="symptom-guide">
            <h4>📸 Image Upload Guidelines:</h4>
            <ul>
                <li>Upload a clear, well-lit photo</li>
                <li>Make sure the affected area is visible</li>
                <li>Include multiple angles if needed</li>
                <li>Avoid blurry or dark images</li>
                <li>Maximum image size: 10MB</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload an image of the symptoms", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            try:
                # Read the image
                image = Image.open(uploaded_file)
                
                # Resize image if too large (max 1024x1024)
                max_size = 1024
                if max(image.size) > max_size:
                    ratio = max_size / max(image.size)
                    new_size = tuple(int(dim * ratio) for dim in image.size)
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Display the image
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                if st.button("Analyze Image", key="image_analyze"):
                    with st.spinner("Analyzing image..."):
                        try:
                            # Convert image to bytes
                            img_byte_arr = io.BytesIO()
                            image.save(img_byte_arr, format='JPEG', quality=95)
                            img_bytes = img_byte_arr.getvalue()
                            
                            # Convert to base64
                            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                            
                            # Analyze image with Clarifai
                            analysis_results = analyze_animal_image(img_base64)
                            
                            if analysis_results['success']:
                                # Show detected features
                                st.write("🔍 Detected Features:")
                                for result in analysis_results['results']:
                                    st.write(f"- {result['name']}: {result['confidence']}% confidence")
                                
                                # Get health assessment
                                diagnosis = get_health_assessment(analysis_results)
                                process_diagnosis_result(diagnosis, animal_type)
                            else:
                                st.error(f"Image analysis failed: {analysis_results['error']}")
                                
                        except Exception as e:
                            st.error(f"An error occurred during image analysis: {str(e)}")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

# Vet Finder Tab
with tab2:
    st.header("🔍 Find a Veterinarian")
    
    col1, col2 = st.columns(2)
    
    with col1:
        location = st.text_input("Location", placeholder="Enter city name (e.g., Hyderabad)")
    
    with col2:
        animal_type = st.selectbox(
            "Animal Type",
            options=["cattle", "buffalo", "sheep", "goat", "poultry"]
        )
    
    if st.button("Search", key="search_button"):
        if location:
            with st.spinner("Searching for veterinarians..."):
                try:
                    vets = find_nearby_vets(location=location, animal_type=animal_type)
                    
                    if vets:
                        st.success(f"✅ Found {len(vets)} veterinarians in {location} for {animal_type}")
                        for vet in vets:
                            st.markdown(f"""
                            <div class="vet-card">
                                <h4>👨‍⚕️ {vet["name"]}</h4>
                                <p>📞 Contact: {vet["contact"]}</p>
                                <p>📍 Address: {vet["address"]}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning(f"No veterinarians found in {location} for {animal_type}")
                        st.markdown("""
                        <div class="symptom-guide">
                            <h4>🔍 Try:</h4>
                            <ul>
                                <li>Checking the spelling of the location</li>
                                <li>Searching in a nearby larger city</li>
                                <li>Selecting a different animal type</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a location to search.")

# Footer
st.markdown("---")
st.markdown("### 📱 Need immediate assistance?")
st.markdown("Use our WhatsApp bot for quick diagnosis and vet finding:")
st.code("1. Save +917483255225 as BioGuardian\n2. Send 'symptoms: [describe symptoms]' for diagnosis\n3. Send 'find vet: [location] for [animal]' to find vets", language="markdown")

