import os
import sys
import json
import argparse
from dotenv import load_dotenv
import speech_recognition as sr
from langchain_groq import ChatGroq
from PIL import Image
import io
import base64
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
import re

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("Error: GROQ_API_KEY not found in .env file", file=sys.stderr)
    sys.exit(1)

# Initialize clients
groq_client = ChatGroq(
    api_key=api_key,
    model="llama-3.1-8b-instant"
)
clarifai_channel = ClarifaiChannel.get_grpc_channel()
clarifai_stub = service_pb2_grpc.V2Stub(clarifai_channel)
CLARIFAI_PAT = os.getenv("CLARIFAI_API_KEY", "")  # Optional: Set your Clarifai PAT for higher limits

def speech_to_text(audio_data):
    """Convert speech to text"""
    recognizer = sr.Recognizer()
    try:
        # Convert audio data to WAV file object
        audio_file = io.BytesIO(audio_data)
        
        # Adjust the recognizer parameters
        recognizer.energy_threshold = 300  # Increase if too sensitive
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8  # Adjust for shorter/longer pauses
        
        try:
            with sr.AudioFile(audio_file) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Record audio from file
                audio = recognizer.record(source)
                # Try to recognize the speech
                text = recognizer.recognize_google(audio)
                print(f"Recognized text: {text}")  # Debug print
                return text
        except sr.UnknownValueError:
            print("Speech recognition could not understand the audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from speech recognition service: {str(e)}")
            return None
            
    except Exception as e:
        print(f"Speech recognition error: {str(e)}")
        return None

def analyze_image(image_data):
    """Analyze image using Clarifai's general model"""
    try:
        # Prepare the image for Clarifai
        metadata = (('authorization', f'Key {CLARIFAI_PAT}'),) if CLARIFAI_PAT else None
        
        # Create the request
        request = service_pb2.PostModelOutputsRequest(
            model_id='general-image-recognition',
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(
                            base64=image_data
                        )
                    )
                )
            ]
        )
        
        # Make the request
        response = clarifai_stub.PostModelOutputs(request, metadata=metadata)
        
        # Check for successful response
        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(f"Request failed, status: {response.status.description}")
        
        # Extract relevant animal and health-related concepts
        relevant_labels = [
            concept.name
            for output in response.outputs
            for concept in output.data.concepts
            if any(keyword in concept.name.lower() 
                  for keyword in ['animal', 'cattle', 'buffalo', 'sheep', 'goat', 'wound', 'swelling', 'injury'])
            and concept.value > 0.5  # Only include concepts with confidence > 50%
        ]
        
        return relevant_labels
    except Exception as e:
        print(f"Image analysis error: {str(e)}")
        return []

def get_recommendation(condition: str, confidence: int) -> dict:
    # Always recommend vet for unknown conditions or very low confidence
    if condition.lower() == "unknown" or confidence < 30:
        return {
            "condition": condition,
            "confidence": confidence,
            "recommendation": "Due to unclear or complex symptoms, immediate veterinary consultation is recommended for proper diagnosis and treatment.",
            "needs_vet": True
        }
    elif confidence < 60:
        return {
            "condition": condition,
            "confidence": confidence,
            "recommendation": "Please consult a veterinarian as the symptoms need professional evaluation.",
            "needs_vet": True
        }
    else:
        recommendations = {
            "mastitis": "1. Isolate the affected animal\n2. Clean udder with warm water\n3. Apply mastitis ointment\n4. Monitor milk production",
            "bloat": "1. Stop feeding immediately\n2. Keep animal standing\n3. Walk the animal slowly\n4. Massage left side of abdomen",
            "fever": "1. Provide plenty of water\n2. Keep in shade\n3. Monitor temperature\n4. Reduce feed intake",
            "hoof infection": "1. Clean the hoof area\n2. Apply antiseptic solution\n3. Keep area dry\n4. Limit movement",
            "unknown": "Monitor the animal closely and record any changes in symptoms"
        }
        # Convert condition to lowercase for matching
        condition_lower = condition.lower()
        recommendation = recommendations.get(condition_lower, "Monitor and record symptoms")
        return {
            "condition": condition,
            "confidence": confidence,
            "recommendation": recommendation,
            "needs_vet": False
        }

def analyze_symptom_clarity(text: str, animal_type: str) -> dict:
    """Analyze if the symptom description is clear enough for diagnosis"""
    # Common symptoms and conditions for quick matching
    clear_symptoms = {
        'fever': [
            'fever', 'high temperature', 'hot', 'warm', 'not eating',
            'lethargic', 'weak', 'reduced appetite', 'drinking more'
        ],
        'mastitis': [
            'udder swollen', 'udder hard', 'milk production down',
            'abnormal milk', 'hot udder', 'painful udder', 'watery milk',
            'clumpy milk', 'blood in milk', 'reduced milk'
        ],
        'bloat': [
            'swollen belly', 'distended stomach', 'not ruminating',
            'kicking at belly', 'difficulty breathing', 'discomfort',
            'enlarged left side', 'restless', 'rapid breathing'
        ],
        'hoof_infection': [
            'limping', 'lame', 'hoof swollen', 'difficulty walking',
            'hot hoof', 'swollen leg', 'not bearing weight', 'hoof damage',
            'foot rot', 'foul smell'
        ]
    }
    
    # Check word count and detail level
    words = text.lower().split()
    word_count = len(words)
    
    # Check for timing indicators
    timing_words = ['yesterday', 'days', 'weeks', 'hours', 'since', 'started',
                   'morning', 'evening', 'today', 'ago', 'noticed']
    has_timing = any(word in text.lower() for word in timing_words)
    
    # Check for specific measurements or observations
    specific_indicators = [
        'temperature', 'degrees', 'swollen', 'discharge', 'color',
        'size', 'weight', 'amount', 'frequency', 'rate', 'pressure',
        'measurement', 'level', 'quantity', 'quality'
    ]
    has_specifics = any(word in text.lower() for word in specific_indicators)
    
    # Check for behavioral descriptions
    behavioral_indicators = [
        'eating', 'drinking', 'walking', 'standing', 'lying',
        'behavior', 'movement', 'appetite', 'activity', 'mood',
        'energy', 'response', 'reaction', 'attitude'
    ]
    has_behavioral = any(word in text.lower() for word in behavioral_indicators)
    
    # Check for clear symptom matches
    matched_symptoms = []
    symptom_count = 0
    for condition, indicators in clear_symptoms.items():
        condition_matches = []
        for indicator in indicators:
            if indicator in text.lower():
                condition_matches.append(indicator)
                symptom_count += 1
        if condition_matches:
            matched_symptoms.append({
                'condition': condition,
                'matches': condition_matches
            })
    
    # Determine if the description is clear enough
    is_clear = (
        # Case 1: Detailed description with timing and specifics
        (word_count >= 10 and has_timing and has_specifics) or
        # Case 2: Long description with behavior and either timing or specifics
        (word_count >= 15 and has_behavioral and (has_timing or has_specifics)) or
        # Case 3: Multiple clear symptoms (increased threshold)
        (symptom_count >= 3) or
        # Case 4: Very detailed description with at least one clear symptom match
        (len(matched_symptoms) > 0 and word_count >= 20) or
        # Case 5: Extremely detailed description
        (word_count >= 30 and (has_timing or has_specifics or has_behavioral)) or
        # Case 6: Multiple symptom matches with timing or specifics
        (len(matched_symptoms) >= 2 and (has_timing or has_specifics))
    )
    
    # If we have multiple strong indicators, override follow-up
    strong_description = (
        (symptom_count >= 4) or  # Many matching symptoms
        (len(matched_symptoms) >= 2 and has_timing and has_specifics) or  # Multiple conditions with details
        (word_count >= 40 and has_timing and has_specifics and has_behavioral)  # Very detailed description
    )
    
    return {
        'is_clear': is_clear or strong_description,
        'matched_symptoms': matched_symptoms,
        'needs_followup': not (is_clear or strong_description),
        'details': {
            'word_count': word_count,
            'has_timing': has_timing,
            'has_specifics': has_specifics,
            'has_behavioral': has_behavioral,
            'symptom_count': symptom_count
        }
    }

def diagnose_symptoms(input_data, input_type='text', animal_type=None) -> dict:
    """
    Diagnose symptoms from various input types
    input_type: 'text', 'voice', or 'image'
    """
    try:
        print(f"\nDiagnosing symptoms:")
        print(f"Input type: {input_type}")
        print(f"Animal type: {animal_type}")
        print(f"Input data: {input_data}")

        # Process input based on type
        if input_type == 'voice':
            text = speech_to_text(input_data)
            print(f"Converted voice to text: {text}")
            if not text:
                return {
                    "condition": "unknown",
                    "confidence": 0,
                    "recommendation": "Could not understand the audio. Please try again or use text input.",
                    "needs_vet": False,
                    "needs_followup": True
                }
        elif input_type == 'image':
            labels = analyze_image(input_data)
            print(f"Analyzed image labels: {labels}")
            if not labels:
                return {
                    "condition": "unknown",
                    "confidence": 0,
                    "recommendation": "Could not analyze the image. Please try again or use text input.",
                    "needs_vet": False,
                    "needs_followup": True
                }
            text = f"Visual symptoms: {', '.join(labels)}"
        else:  # text input
            # Remove any extra quotes that might have been added
            text = input_data.strip('"')
            print(f"Processed text input: {text}")

        # Add animal type to context if provided
        if animal_type:
            text = f"Animal type: {animal_type}. Symptoms: {text}"
        
        print(f"Final text for analysis: {text}")

        # Analyze symptom clarity
        clarity_analysis = analyze_symptom_clarity(text, animal_type)
        print(f"Clarity analysis: {json.dumps(clarity_analysis, indent=2)}")
        if clarity_analysis['needs_followup']:
            return {
                "condition": "unclear",
                "confidence": 0,
                "recommendation": "Please provide more specific information about the symptoms.",
                "needs_vet": False,
                "needs_followup": True,
                "matched_symptoms": clarity_analysis['matched_symptoms']
            }

        messages = [
            {"role": "system", "content": """You are a veterinary diagnostic assistant specialized in livestock health. Follow these rules:
1. Analyze symptoms carefully and match them against known conditions
2. Consider the animal type and specific symptoms when making a diagnosis
3. Return 'unknown' ONLY if symptoms are truly unclear or don't match any known condition
4. Assign confidence levels based on:
   - Symptom specificity (more specific = higher confidence)
   - Number of matching symptoms
   - Presence of definitive indicators
5. Use this confidence scale:
   - 80-100%: Clear, definitive symptoms matching a specific condition
   - 60-79%: Strong indicators but some symptoms might overlap
   - 30-59%: Moderate indicators but needs more information
   - 0-29%: Unclear or conflicting symptoms"""},
            {"role": "user", "content": f"""
Based on the symptoms described, classify the likely condition.

Symptoms: "{text}"

Consider these conditions and their key indicators:
- mastitis: udder swelling, abnormal milk, reduced production, warmth
- bloat: swollen belly, difficulty breathing, discomfort, not ruminating
- fever: high temperature, lethargy, reduced appetite, warm body
- hoof infection: limping, swelling, difficulty walking, hoof damage
- unknown: if symptoms don't clearly match any condition

Respond in JSON format with condition and confidence (0-100):
{{"condition": "condition_name", "confidence": confidence_level}}

Example responses:
{{"condition": "mastitis", "confidence": 85}} - for clear mastitis symptoms
{{"condition": "unknown", "confidence": 0}} - for unclear symptoms"""}
        ]

        print("Sending request to Groq API...")
        response = groq_client.invoke(messages)
        response_text = response.content.strip()
        print(f"Received response: {response_text}")
        
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'{.*}', response_text)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # If no JSON found, try to parse the entire response
                result = json.loads(response_text)
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Error parsing response: {str(e)}")
            print("Response text:", response_text)
            # Default to unknown only if parsing fails
            result = {"condition": "unknown", "confidence": 0}
        
        # Validate the response format
        if not isinstance(result.get("confidence"), (int, float)):
            print("Invalid confidence value in response")
            result["confidence"] = 0
        
        if not isinstance(result.get("condition"), str):
            print("Invalid condition value in response")
            result["condition"] = "unknown"
        
        # Get recommendation
        diagnosis = get_recommendation(result["condition"], result["confidence"])
        diagnosis["needs_followup"] = False  # Add this field to indicate no more follow-up needed
        
        return diagnosis

    except Exception as e:
        print(f"Error in diagnose_symptoms: {str(e)}")
        return {
            "condition": "unknown",
            "confidence": 0,
            "recommendation": "An error occurred during diagnosis. Please try again or consult a veterinarian.",
            "needs_vet": True,
            "needs_followup": True
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diagnose livestock symptoms')
    parser.add_argument('--input_data', required=True, help='Input data (text, base64 image, or audio)')
    parser.add_argument('--input_type', default='text', choices=['text', 'voice', 'image'], help='Type of input')
    parser.add_argument('--animal_type', help='Type of animal (optional)')
    
    try:
        args = parser.parse_args()
        print(f"\nAnalyzing {args.input_type} input for {args.animal_type or 'unspecified animal'}\n", file=sys.stderr)
        
        result = diagnose_symptoms(
            input_data=args.input_data,
            input_type=args.input_type,
            animal_type=args.animal_type
        )
        
        # Always output the result as JSON to stdout
        print(json.dumps(result))
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        # Return a JSON error response
        print(json.dumps({
            "condition": "unknown",
            "confidence": 0,
            "recommendation": f"An error occurred: {str(e)}",
            "needs_vet": True,
            "needs_followup": True
        }))
